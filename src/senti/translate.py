"""Machine translate the fresh 2022 to 2023 Amazon reviews into other languages.

No public multilingual sentiment test set is safe from the multilingual models' training data
(see overlap.py), so the same 1,200 fresh English reviews are translated with NLLB-200 and keep
their star rating labels. Title and body are translated sentence by sentence, then rejoined.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

MT_MODEL = "facebook/nllb-200-distilled-600M"
LANGS = {
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ja": "jpn_Jpan",
    "zh": "zho_Hans",
}


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(text))
    return [p.strip() for p in parts if p.strip()]


def split_review(text: str) -> tuple[str, str]:
    title, _, body = str(text).partition("\n\n")
    return title.strip(), body.strip()


class Translator:
    def __init__(self, target: str, model: str = MT_MODEL, batch_size: int = 24, beams: int = 2):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, src_lang="eng_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model).eval()
        self.target_id = self.tok.convert_tokens_to_ids(LANGS[target])
        self.batch_size, self.beams = batch_size, beams

    def __call__(self, texts: list[str]) -> list[str]:
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [""] * len(texts)
        with self.torch.inference_mode():
            for s in range(0, len(order), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=200)
                gen = self.model.generate(
                    **enc, forced_bos_token_id=self.target_id, num_beams=self.beams, max_new_tokens=int(enc["input_ids"].shape[1] * 2 + 10)
                )
                for i, t in zip(idx, self.tok.batch_decode(gen, skip_special_tokens=True), strict=True):
                    out[i] = t
        return out


def translate_reviews(texts: list[str], translate) -> list[str]:
    """Translate every sentence once, then rebuild each review as title, blank line, body."""
    pieces, slots = [], []
    for r, text in enumerate(texts):
        title, body = split_review(text)
        for part, chunk in ((0, title), (1, body)):
            for s in sentences(chunk):
                slots.append((r, part))
                pieces.append(s)
    done = translate(pieces)
    rebuilt = [[[], []] for _ in texts]
    for (r, part), t in zip(slots, done, strict=True):
        rebuilt[r][part].append(t)
    return [(" ".join(t) + ("\n\n" + " ".join(b) if b else "")).strip() for t, b in rebuilt]


def run(data: Path, lang: str, out: Path) -> pd.DataFrame:
    df = pd.read_parquet(data / "samples.parquet")
    src = df[df["set"] == "amazon23"].reset_index(drop=True)
    texts = translate_reviews(src["text"].tolist(), Translator(lang))
    res = src.assign(text=texts, lang=lang, set="amazon23_mt", id=src["id"] + f"_mt_{lang}")
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"mt_{lang}.parquet", index=False)
    print(lang, len(res), "example:", texts[0][:200], flush=True)
    return res


def merge(data: Path, mt_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(data / "samples.parquet")
    parts = [pd.read_parquet(f) for f in sorted(mt_dir.glob("mt_*.parquet"))]
    df = pd.concat([df[df["set"] != "amazon23_mt"], *parts], ignore_index=True)
    assert df["id"].is_unique
    df.to_parquet(data / "samples.parquet", index=False)
    print(df.groupby(["set", "split"]).size(), flush=True)
    return df


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        run(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
