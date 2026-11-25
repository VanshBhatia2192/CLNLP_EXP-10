"""Train a model and write its translations of the test set to outputs/predictions/.

    python train.py --model gru
    python train.py --model transformer
    python train.py --model nllb              # NLLB-200 zero-shot (no training)
    python train.py --model nllb --finetune   # zero-shot, then fine-tune (needs a CUDA GPU)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import sacrebleu
import torch
import torch.nn as nn

import data
import models
from data import OUTPUT_DIR, PAD, PRED_DIR

LABEL_SMOOTHING = 0.1
NLLB_FT_DIR = os.path.join(OUTPUT_DIR, "nllb_finetuned")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def checkpoint_path(name):
    return os.path.join(OUTPUT_DIR, f"{name}.pt")


def write_predictions(name, hyps):
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(hyps) + "\n")
    print(f"Test translations written to {path}")


# ---------------- scratch models ----------------
@torch.no_grad()
def translate(model, src_ids, sp_tgt, device, batch_size=128):
    """Greedy-decode a list of encoded source sentences."""
    model.eval()
    order = sorted(range(len(src_ids)), key=lambda i: len(src_ids[i]))
    hyps = [None] * len(src_ids)
    for b in range(0, len(order), batch_size):
        ids = order[b:b + batch_size]
        out = model.greedy_decode(data.pad([src_ids[i] for i in ids]).to(device), data.MAX_LEN + 10)
        for j, i in enumerate(ids):
            hyps[i] = data.decode(sp_tgt, out[j].tolist())
    return hyps


def run_epoch(model, src_ids, tgt_ids, crit, device, batch_tokens, opt=None, sched=None):
    """One pass over the data: trains when an optimizer is given, otherwise only computes the loss."""
    model.train(opt is not None)
    total, ntok = 0.0, 0
    for bidx in data.make_batches(src_ids, tgt_ids, batch_tokens, shuffle=opt is not None):
        src = data.pad([src_ids[i] for i in bidx]).to(device)
        tgt = data.pad([tgt_ids[i] for i in bidx]).to(device)
        with torch.set_grad_enabled(opt is not None):
            logits = model(src, tgt[:, :-1])   # teacher forcing
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
        k = (tgt[:, 1:] != PAD).sum().item()
        total += loss.item() * k
        ntok += k
    return total / ntok


def train_scratch(name, epochs, device):
    sp_en, sp_as = data.load_tokenizers()
    train_df, dev_df, test_df = (data.load_split(s) for s in ("train", "dev", "test"))
    train_src, train_tgt = data.encode(sp_en, train_df["src"]), data.encode(sp_as, train_df["tgt"])
    dev_src, dev_tgt = data.encode(sp_en, dev_df["src"]), data.encode(sp_as, dev_df["tgt"])

    cfg = dict(models.GRU if name == "gru" else models.TRANSFORMER)
    if epochs:
        cfg["epochs"] = epochs
    model = models.build_model(name, sp_en.get_piece_size(), sp_as.get_piece_size()).to(device)
    print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9)
    sched = None
    if cfg.get("warmup"):  # Noam schedule; the base lr acts as a multiplier
        d, w = model.d_model, cfg["warmup"]
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (d ** -0.5) * min((s + 1) ** -0.5, (s + 1) * w ** -1.5))
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=LABEL_SMOOTHING)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_BLEU": []}
    best, bad = -1.0, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        tr_loss = run_epoch(model, train_src, train_tgt, crit, device, cfg["batch_tokens"], opt, sched)
        dev_loss = run_epoch(model, dev_src, dev_tgt, crit, device, cfg["batch_tokens"])
        dev_bleu = sacrebleu.corpus_bleu(translate(model, dev_src, sp_as, device), [dev_df["tgt"].tolist()]).score
        for k, v in zip(history, [ep, tr_loss, dev_loss, dev_bleu]):
            history[k].append(v)
        print(f"[{name}] epoch {ep:2d} | train loss {tr_loss:.3f} | dev loss {dev_loss:.3f} "
              f"| dev BLEU {dev_bleu:5.2f} | {time.time() - t0:.0f}s")
        if dev_bleu > best:  # early stopping on dev BLEU
            best, bad = dev_bleu, 0
            torch.save(model.state_dict(), checkpoint_path(name))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print(f"Early stopping: no dev BLEU gain for {cfg['patience']} epochs")
                break

    model.load_state_dict(torch.load(checkpoint_path(name), map_location=device))
    with open(os.path.join(OUTPUT_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f)
    write_predictions(name, translate(model, data.encode(sp_en, test_df["src"]), sp_as, device))


# ---------------- NLLB-200 ----------------
def run_nllb(finetune, device):
    test_src = data.load_split("test")["src"].tolist()
    tok, model = models.nllb_load(device=device)
    print(f"NLLB parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")
    write_predictions("nllb_zeroshot", models.nllb_translate(model, tok, test_src, device))

    if finetune:
        if device.type != "cuda":
            print("Warning: fine-tuning a 600M-parameter model without a CUDA GPU will be very slow.")
        model = models.nllb_finetune(model, tok, data.load_split("train"), device)
        model.save_pretrained(NLLB_FT_DIR)
        tok.save_pretrained(NLLB_FT_DIR)
        print(f"Fine-tuned model saved to {NLLB_FT_DIR}")
        write_predictions("nllb_finetuned", models.nllb_translate(model, tok, test_src, device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gru", "transformer", "nllb"], required=True)
    ap.add_argument("--epochs", type=int, help="override the number of epochs (scratch models)")
    ap.add_argument("--finetune", action="store_true", help="fine-tune NLLB after zero-shot evaluation")
    args = ap.parse_args()

    random.seed(data.SEED); np.random.seed(data.SEED); torch.manual_seed(data.SEED)
    device = get_device()
    print("Device:", device)
    if args.model == "nllb":
        run_nllb(args.finetune, device)
    else:
        train_scratch(args.model, args.epochs, device)


if __name__ == "__main__":
    main()
