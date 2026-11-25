"""Data preparation: cleaning, filtering, splitting and SentencePiece tokenisation.

    python data.py
"""
import os
import random
import re
import unicodedata

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch

# ---------------- settings ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATA = os.path.join(ROOT, "data", "raw", "en_as_wmt2026.csv")
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
OUTPUT_DIR = os.path.join(ROOT, "outputs")          # checkpoints, tokenisers, predictions (git-ignored)
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULTS_DIR = os.path.join(ROOT, "results")

SEED = 42
MAX_WORDS = 80            # drop sentences longer than this
MAX_LEN_RATIO = 3.0       # drop pairs whose word-length ratio is above this
MIN_BENGALI_RATIO = 0.8   # share of Assamese-side letters that must be Bengali script
TEST_SIZE = DEV_SIZE = 1000
VOCAB_SIZE = 8000         # SentencePiece vocabulary per language
MAX_LEN = 64              # max subword tokens per sentence (incl. <s> and </s>)
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# ---------------- cleaning ----------------
ZW_RE = re.compile(r"[\u200b\u200e\u200f\ufeff\xa0]")
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"\s+([.,!?;:।)\]’”])")
OPEN_RE = re.compile(r"([(\[‘“])\s+")


def clean_text(s):
    """Unicode-normalise and tidy one sentence."""
    s = unicodedata.normalize("NFC", str(s))
    s = ZW_RE.sub(" ", s)
    # remove control characters but keep zero-width (non-)joiners used in Bengali script
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\u200c\u200d")
    s = SPACE_RE.sub(" ", s).strip()
    s = PUNCT_RE.sub(r"\1", s)   # "word ." -> "word."
    s = OPEN_RE.sub(r"\1", s)    # "( word" -> "(word"
    return s


def bengali_ratio(s):
    letters = [c for c in s if c.isalpha()]
    return sum("\u0980" <= c <= "\u09ff" for c in letters) / len(letters) if letters else 0.0


def preprocess(df):
    """Clean, filter and de-duplicate. Returns the clean DataFrame and row counts after each step."""
    stats = {"raw": len(df)}
    df = df.dropna().copy()
    df["src"] = df["src"].map(clean_text)
    df["tgt"] = df["tgt"].map(clean_text)
    df = df[(df["src"].str.len() > 0) & (df["tgt"].str.len() > 0)]
    stats["after_empty_filter"] = len(df)

    sl, tl = df["src"].str.split().str.len(), df["tgt"].str.split().str.len()
    df = df[(sl <= MAX_WORDS) & (tl <= MAX_WORDS)]
    sl, tl = df["src"].str.split().str.len(), df["tgt"].str.split().str.len()
    df = df[(np.maximum(sl, tl) / np.minimum(sl, tl)) <= MAX_LEN_RATIO]
    stats["after_length_filter"] = len(df)

    df = df[df["tgt"].map(bengali_ratio) >= MIN_BENGALI_RATIO]
    stats["after_script_filter"] = len(df)

    df = df.drop_duplicates(subset=["src", "tgt"]).drop_duplicates(subset=["src"])
    stats["after_dedup"] = len(df)
    return df.reset_index(drop=True), stats


def split(df):
    """Seeded random split into train / dev / test."""
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_test = min(TEST_SIZE, int(0.05 * len(df)))
    n_dev = min(DEV_SIZE, int(0.05 * len(df)))
    test = df.iloc[:n_test].reset_index(drop=True)
    dev = df.iloc[n_test:n_test + n_dev].reset_index(drop=True)
    train = df.iloc[n_test + n_dev:].reset_index(drop=True)
    return train, dev, test


def load_split(name):
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - run `python data.py` first.")
    return pd.read_csv(path, keep_default_na=False)


# ---------------- SentencePiece tokenisation ----------------
SPM_DIR = os.path.join(OUTPUT_DIR, "tokenizers")


def train_spm(sentences, name):
    """Train a unigram SentencePiece model on the given sentences."""
    os.makedirs(SPM_DIR, exist_ok=True)
    prefix = os.path.join(SPM_DIR, name)
    with open(prefix + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))
    spm.SentencePieceTrainer.train(
        input=prefix + ".txt", model_prefix=prefix, vocab_size=VOCAB_SIZE, model_type="unigram",
        character_coverage=1.0, pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, minloglevel=2)
    os.remove(prefix + ".txt")
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def load_tokenizers():
    """Returns the (English, Assamese) SentencePiece models."""
    sps = []
    for name in ("spm_en", "spm_as"):
        path = os.path.join(SPM_DIR, name + ".model")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run `python data.py` first.")
        sps.append(spm.SentencePieceProcessor(model_file=path))
    return tuple(sps)


def encode(sp, sentences):
    """Sentences -> lists of ids wrapped in <s> ... </s> and truncated to MAX_LEN."""
    return [[BOS] + ids[:MAX_LEN - 2] + [EOS] for ids in sp.encode(list(sentences))]


def decode(sp, ids):
    out = []
    for t in ids:
        if t == EOS:
            break
        if t not in (PAD, BOS):
            out.append(t)
    return sp.decode(out)


def pad(seqs):
    m = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (m - len(s)) for s in seqs], dtype=torch.long)


def make_batches(src, tgt, batch_tokens, shuffle=True):
    """Group sentences of similar length so that each batch has about `batch_tokens` tokens."""
    idx = sorted(range(len(src)), key=lambda i: (len(src[i]), len(tgt[i])))
    batches, cur, cur_max = [], [], 0
    for i in idx:
        length = max(len(src[i]), len(tgt[i]))
        if cur and max(cur_max, length) * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, length)
    if cur:
        batches.append(cur)
    if shuffle:
        random.shuffle(batches)
    return batches


# ---------------- main: prepare everything ----------------
def main():
    raw = pd.read_csv(RAW_DATA).iloc[:, :2]
    raw.columns = ["src", "tgt"]
    print(f"Loaded {len(raw):,} raw sentence pairs")

    clean, stats = preprocess(raw)
    print("\nRows after each cleaning step:\n" + pd.Series(stats).to_string())

    train, dev, test = split(clean)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    for name, d in [("train", train), ("dev", dev), ("test", test)]:
        d.to_csv(os.path.join(PROCESSED_DIR, f"{name}.csv"), index=False)
    print(f"\ntrain: {len(train):,}  dev: {len(dev):,}  test: {len(test):,}  -> saved to {PROCESSED_DIR}")

    sp_en = train_spm(train["src"].tolist(), "spm_en")
    sp_as = train_spm(train["tgt"].tolist(), "spm_as")
    lens = pd.Series([len(x) for x in encode(sp_as, train["tgt"])])
    print(f"\nSentencePiece vocab: EN {sp_en.get_piece_size()}  AS {sp_as.get_piece_size()}")
    print(f"Avg Assamese subwords per sentence: {lens.mean():.1f} "
          f"(truncated at {MAX_LEN}: {(lens >= MAX_LEN).mean() * 100:.2f}%)")

    ex = train.iloc[0]
    print("\nExample:")
    print("  EN:", ex["src"], "\n     ", sp_en.encode(ex["src"], out_type=str))
    print("  AS:", ex["tgt"], "\n     ", sp_as.encode(ex["tgt"], out_type=str))


if __name__ == "__main__":
    main()
