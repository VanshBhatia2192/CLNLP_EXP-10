"""Models: Bi-GRU + Bahdanau attention and Transformer (trained from scratch), and pre-trained NLLB-200."""
import gc
import math
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import BOS, EOS, PAD, SEED, clean_text

# ---------------- hyper-parameters ----------------
GRU = dict(emb=256, hid=256, dropout=0.3,
           epochs=12, lr=1e-3, batch_tokens=5000, patience=3)
TRANSFORMER = dict(d_model=256, nhead=4, layers=3, ff=1024, dropout=0.2,
                   epochs=25, lr=0.7, warmup=800, batch_tokens=4000, patience=4)

NLLB_NAME = "facebook/nllb-200-distilled-600M"
NLLB_SRC, NLLB_TGT = "eng_Latn", "asm_Beng"
NUM_BEAMS = 4
FT_EPOCHS, FT_LR = 1, 1e-4
FT_BATCH, FT_GRAD_ACCUM = 8, 4   # effective batch = 32
FT_MAX_LEN = 96
FT_MAX_SENTENCES = None          # e.g. 20000 for a quicker fine-tuning run


# ================= scratch models =================


class Seq2SeqAttn(nn.Module):
    """Bidirectional GRU encoder + GRU decoder with additive (Bahdanau) attention."""

    def __init__(self, src_vocab, tgt_vocab, emb=256, hid=256, dropout=0.3):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, emb, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb, padding_idx=PAD)
        self.encoder = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hid, 2 * hid)
        self.decoder = nn.GRUCell(emb + 2 * hid, 2 * hid)
        self.attn_W = nn.Linear(2 * hid, 2 * hid, bias=False)
        self.attn_U = nn.Linear(2 * hid, 2 * hid, bias=False)
        self.attn_v = nn.Linear(2 * hid, 1, bias=False)
        self.out = nn.Linear(2 * hid + 2 * hid + emb, tgt_vocab)
        self.drop = nn.Dropout(dropout)

    def encode(self, src):
        mask = src != PAD
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.drop(self.src_emb(src)), lengths,
                                                   batch_first=True, enforce_sorted=False)
        enc, _ = self.encoder(packed)
        enc, _ = nn.utils.rnn.pad_packed_sequence(enc, batch_first=True, total_length=src.size(1))
        mean = (enc * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        h0 = torch.tanh(self.bridge(mean))           # initial decoder state
        return enc, mask, h0, self.attn_U(enc)

    def step(self, y, h, enc, mask, enc_proj):
        e = self.drop(self.tgt_emb(y))
        scores = self.attn_v(torch.tanh(self.attn_W(h).unsqueeze(1) + enc_proj)).squeeze(-1)
        attn = torch.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
        ctx = torch.bmm(attn.unsqueeze(1), enc).squeeze(1)
        h = self.decoder(torch.cat([e, ctx], -1), h)
        logits = self.out(self.drop(torch.cat([h, ctx, e], -1)))
        return logits, h

    def forward(self, src, tgt_in):
        enc, mask, h, enc_proj = self.encode(src)
        outs = []
        for t in range(tgt_in.size(1)):        # teacher forcing
            logits, h = self.step(tgt_in[:, t], h, enc, mask, enc_proj)
            outs.append(logits)
        return torch.stack(outs, 1)

    @torch.no_grad()
    def greedy_decode(self, src, max_len):
        enc, mask, h, enc_proj = self.encode(src)
        y = torch.full((src.size(0),), BOS, dtype=torch.long, device=src.device)
        out, done = [], torch.zeros_like(y, dtype=torch.bool)
        for _ in range(max_len):
            logits, h = self.step(y, h, enc, mask, enc_proj)
            y = logits.argmax(-1)
            out.append(y)
            done |= y == EOS
            if done.all():
                break
        return torch.stack(out, 1)


class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerMT(nn.Module):
    """Small pre-LayerNorm Transformer encoder-decoder with tied output embeddings."""

    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=4, layers=3, ff=1024, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, nhead, layers, layers, ff, dropout,
                                          batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, tgt_vocab, bias=False)
        self.out.weight = self.tgt_emb.weight
        for p in self.transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.src_emb.weight, 0, d_model ** -0.5)
        nn.init.normal_(self.tgt_emb.weight, 0, d_model ** -0.5)

    def embed(self, emb, x):
        return self.drop(self.pos(emb(x) * math.sqrt(self.d_model)))

    def encode(self, src):
        src_pad = src == PAD
        mem = self.transformer.encoder(self.embed(self.src_emb, src), src_key_padding_mask=src_pad)
        return mem, src_pad

    def decode(self, tgt_in, mem, src_pad):
        n = tgt_in.size(1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tgt_in.device), 1)
        h = self.transformer.decoder(self.embed(self.tgt_emb, tgt_in), mem, tgt_mask=causal,
                                     tgt_key_padding_mask=(tgt_in == PAD),
                                     memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        mem, src_pad = self.encode(src)
        return self.decode(tgt_in, mem, src_pad)

    @torch.no_grad()
    def greedy_decode(self, src, max_len):
        mem, src_pad = self.encode(src)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            nxt = self.decode(ys, mem, src_pad)[:, -1].argmax(-1).masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


def build_model(name, src_vocab, tgt_vocab):
    if name == "gru":
        c = GRU
        return Seq2SeqAttn(src_vocab, tgt_vocab, c["emb"], c["hid"], c["dropout"])
    if name == "transformer":
        c = TRANSFORMER
        return TransformerMT(src_vocab, tgt_vocab, c["d_model"], c["nhead"], c["layers"], c["ff"], c["dropout"])
    raise ValueError(f"unknown model '{name}' (use 'gru' or 'transformer')")


# ================= pre-trained NLLB-200 =================
def nllb_load(model_path=NLLB_NAME, device="cpu"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(NLLB_NAME, src_lang=NLLB_SRC, tgt_lang=NLLB_TGT)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tok, model


@torch.no_grad()
def nllb_translate(model, tok, sentences, device, batch_size=32, num_beams=NUM_BEAMS):
    model.eval()
    forced = tok.convert_tokens_to_ids(NLLB_TGT)
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    hyps = [None] * len(sentences)
    for b in range(0, len(order), batch_size):
        ids = order[b:b + batch_size]
        enc = tok([sentences[i] for i in ids], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(device)
        out = model.generate(**enc, forced_bos_token_id=forced, num_beams=num_beams,
                             max_new_tokens=128, max_length=None)
        for j, text in zip(ids, tok.batch_decode(out, skip_special_tokens=True)):
            hyps[j] = clean_text(text)
    return hyps


def nllb_finetune(model, tok, train_df, device):
    """Fine-tune on the WMT training data.

    Fits a 15 GB T4 GPU by using Adafactor (almost no optimizer memory), gradient
    checkpointing, fp16 mixed precision and gradient accumulation.
    """
    from transformers.optimization import Adafactor
    use_amp = device.type == "cuda"
    if FT_MAX_SENTENCES:
        train_df = train_df.sample(FT_MAX_SENTENCES, random_state=SEED)
    pairs = list(zip(train_df["src"], train_df["tgt"]))

    def collate(batch):
        s, t = zip(*batch)
        enc = tok(list(s), text_target=list(t), return_tensors="pt", padding=True,
                  truncation=True, max_length=FT_MAX_LEN)
        enc["labels"][enc["labels"] == tok.pad_token_id] = -100
        return enc

    loader = DataLoader(pairs, batch_size=FT_BATCH, shuffle=True, collate_fn=collate)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    opt = Adafactor(model.parameters(), lr=FT_LR, scale_parameter=False,
                    relative_step=False, warmup_init=False, weight_decay=0.0)
    total = FT_EPOCHS * math.ceil(len(loader) / FT_GRAD_ACCUM)
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 1 - s / total))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    step = 0
    for _ in range(FT_EPOCHS):
        t0, run, n_run = time.time(), 0.0, 0
        opt.zero_grad()
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss / FT_GRAD_ACCUM
            scaler.scale(loss).backward()
            run += loss.item() * FT_GRAD_ACCUM
            n_run += 1
            if (i + 1) % FT_GRAD_ACCUM == 0 or (i + 1) == len(loader):
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
                step += 1
                if step % 50 == 0 or step == total:
                    print(f"step {step}/{total} | loss {run / n_run:.3f} | {time.time() - t0:.0f}s")
                    run, n_run = 0.0, 0

    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    del opt, sched, scaler
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()
    return model
