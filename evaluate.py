"""Score every model's test translations, draw the comparison plots, or translate your own sentences.

    python evaluate.py
    python evaluate.py --translate --model transformer "The farmers are growing rice in the field."
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
import torch

import data
import models
from data import OUTPUT_DIR, PRED_DIR, RESULTS_DIR

MODELS = {  # prediction file -> display name
    "gru": "GRU + Attention (scratch)",
    "transformer": "Transformer (scratch)",
    "nllb_zeroshot": "NLLB-200 600M (zero-shot)",
    "nllb_finetuned": "NLLB-200 600M (fine-tuned)",
}
FIG_DIR = os.path.join(RESULTS_DIR, "figures")


def score(hyps, refs):
    return {
        "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2),
        "chrF++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 2),
        "TER": round(sacrebleu.corpus_ter(hyps, [refs]).score, 2),
    }


def load_predictions(n):
    preds = {}
    for key, name in MODELS.items():
        path = os.path.join(PRED_DIR, f"{key}.txt")
        if os.path.exists(path):
            lines = open(path, encoding="utf-8").read().split("\n")[:n]
            if len(lines) == n:
                preds[name] = lines
    return preds


def plot_metrics(results):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for a, m in zip(ax, ["BLEU", "chrF++", "TER"]):
        vals = results[m]
        bars = a.bar(range(len(vals)), vals, color=colors[:len(vals)])
        a.set_xticks(range(len(vals)))
        a.set_xticklabels([i.replace(" (", "\n(") for i in vals.index], fontsize=8)
        a.set_title(m + (" (lower is better)" if m == "TER" else " (higher is better)"))
        for bar, v in zip(bars, vals):
            a.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"), dpi=120)
    plt.close()


def plot_by_length(test_df, preds):
    buckets = pd.cut(test_df["src"].str.split().str.len(), [0, 10, 20, 30, 200],
                     labels=["1-10", "11-20", "21-30", "30+"])
    rows = []
    for name, hyps in preds.items():
        for b in buckets.cat.categories:
            idx = buckets[buckets == b].index
            if len(idx):
                rows.append({"Model": name, "Length": b,
                             "chrF++": sacrebleu.corpus_chrf([hyps[i] for i in idx], [test_df["tgt"][idx].tolist()], word_order=2).score})
    table = pd.DataFrame(rows).pivot(index="Length", columns="Model", values="chrF++").round(2)
    table.plot(kind="bar", figsize=(10, 4), rot=0, title="chrF++ by English sentence length (words)")
    plt.ylabel("chrF++")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "chrf_by_length.png"), dpi=120)
    plt.close()
    return table


def plot_training_curves():
    hists = []
    for key in ("gru", "transformer"):
        path = os.path.join(OUTPUT_DIR, f"{key}_history.json")
        if os.path.exists(path):
            hists.append((json.load(open(path)), MODELS[key].split(" (")[0]))
    if not hists:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for h, label in hists:
        ax[0].plot(h["epoch"], h["train_loss"], "-o", ms=3, label=f"{label} train")
        ax[0].plot(h["epoch"], h["dev_loss"], "--", label=f"{label} dev")
        ax[1].plot(h["epoch"], h["dev_BLEU"], "-o", ms=3, label=label)
    ax[0].set_title("Label-smoothed cross-entropy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].set_title("Dev BLEU"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "training_curves.png"), dpi=120)
    plt.close()


def translate_sentences(model_name, sentences):
    from train import checkpoint_path, get_device, translate, NLLB_FT_DIR
    device = get_device()
    if model_name in ("gru", "transformer"):
        sp_en, sp_as = data.load_tokenizers()
        model = models.build_model(model_name, sp_en.get_piece_size(), sp_as.get_piece_size()).to(device)
        model.load_state_dict(torch.load(checkpoint_path(model_name), map_location=device))
        hyps = translate(model, data.encode(sp_en, sentences), sp_as, device)
    else:
        path = models.NLLB_NAME if model_name == "nllb" else NLLB_FT_DIR
        tok, model = models.nllb_load(path, device)
        hyps = models.nllb_translate(model, tok, sentences, device)
    for src, hyp in zip(sentences, hyps):
        print(f"EN: {src}\nAS: {hyp}\n")


def evaluate_all():
    test_df = data.load_split("test")
    refs = test_df["tgt"].tolist()
    preds = load_predictions(len(test_df))
    if not preds:
        print(f"No prediction files found in {PRED_DIR}. Run train.py first.")
        return

    os.makedirs(FIG_DIR, exist_ok=True)
    results = pd.DataFrame({name: score(h, refs) for name, h in preds.items()}).T
    results.index.name = "Model"
    results.to_csv(os.path.join(RESULTS_DIR, "results.csv"))
    print(results.to_string(), "\n")

    pd.DataFrame({"source": test_df["src"], "reference": refs, **preds}).to_csv(
        os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)

    plot_metrics(results)
    print("chrF++ by sentence length:\n", plot_by_length(test_df, preds).to_string())
    plot_training_curves()
    print(f"\nSaved results.csv, test_predictions.csv and figures to {RESULTS_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--translate", action="store_true", help="translate the given sentences instead of evaluating")
    ap.add_argument("--model", default="transformer", choices=["gru", "transformer", "nllb", "nllb-finetuned"])
    ap.add_argument("sentences", nargs="*", help="English sentence(s) to translate (with --translate)")
    args = ap.parse_args()
    if args.translate:
        if not args.sentences:
            ap.error("give at least one sentence to translate")
        translate_sentences(args.model, args.sentences)
    else:
        evaluate_all()


if __name__ == "__main__":
    main()
