"""Check how many of our test texts appear word for word in a public model training collection.

clapAI/MultiLingualSentiment is the training data of the clapAI model and contains a large part of
tyqiangz/multilingual-sentiments, the training data of the lxyuan model. A text counts as seen if
its normalised form, or its review body without the title, is in the collection.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

COLLECTION = "clapAI/MultiLingualSentiment"


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def keys(text: str) -> set[str]:
    """Whole text, and the body after a blank line (Amazon texts are title + blank line + body)."""
    out = {norm(text)}
    parts = re.split(r"\n\s*\n", str(text), maxsplit=1)
    if len(parts) == 2 and len(parts[1].strip()) >= 20:
        out.add(norm(parts[1]))
    return {k for k in out if k}


def digest(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def collection_digests(texts, sources=None) -> dict[bytes, int]:
    """digest -> bit mask of the sources (by position in a sources list) the text came from."""
    out: dict[bytes, int] = {}
    sources = [0] * len(texts) if sources is None else sources
    for t, src in zip(texts, sources, strict=True):
        for k in keys(t):
            d = digest(k)
            out[d] = out.get(d, 0) | (1 << src)
    return out


def seen(df: pd.DataFrame, digests: dict[bytes, int]) -> pd.Series:
    return df["text"].map(lambda t: any(digest(k) in digests for k in keys(t)))


def seen_sources(df: pd.DataFrame, digests: dict[bytes, int]) -> pd.Series:
    return df["text"].map(lambda t: _or(digests.get(digest(k), 0) for k in keys(t)))


def _or(masks) -> int:
    m = 0
    for x in masks:
        m |= x
    return m


def run(data: Path, out: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    df = pd.read_parquet(data / "samples.parquet")
    files = [s.rfilename for s in HfApi().dataset_info(COLLECTION).siblings if s.rfilename.endswith(".parquet")]
    result = {"collection": COLLECTION, "by_split": {}}
    source_names: list[str] = []
    mask = pd.Series(0, index=df.index)
    for split in ("train", "validation", "test"):
        frames = [pd.read_parquet(hf_hub_download(COLLECTION, f, repo_type="dataset"), columns=["text", "source"]) for f in files if f"/{split}-" in f]
        if not frames:
            continue
        coll = pd.concat(frames, ignore_index=True)
        for name in coll["source"].astype(str).unique():
            if name not in source_names:
                source_names.append(name)
        code = {n: i for i, n in enumerate(source_names)}
        dig = collection_digests(coll["text"].tolist(), coll["source"].astype(str).map(code).tolist())
        del coll
        m = seen_sources(df, dig)
        df[f"in_{split}"] = m > 0
        mask = mask | m
        result["by_split"][split] = int(df[f"in_{split}"].sum())
    cols = [c for c in df.columns if c.startswith("in_")]
    df["in_collection"] = df[cols].any(axis=1)
    df["sources"] = mask.map(lambda x: ";".join(n for i, n in enumerate(source_names) if x >> i & 1))
    by_source = {}
    for srcs in df.loc[df["in_collection"], "sources"]:
        for n in srcs.split(";"):
            by_source[n] = by_source.get(n, 0) + 1
    result["matches_by_source"] = dict(sorted(by_source.items(), key=lambda kv: -kv[1]))
    table = df.groupby(["set", "lang"])["in_collection"].agg(["sum", "count"]).reset_index()
    result["by_set_lang"] = [{"set": r.set, "lang": r.lang, "seen": int(r["sum"]), "n": int(r["count"])} for _, r in table.iterrows()]
    out.mkdir(parents=True, exist_ok=True)
    df[["id", *cols, "in_collection", "sources"]].to_parquet(out / "overlap.parquet", index=False)
    (out / "overlap.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return result


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1] if len(sys.argv) > 1 else "data"), Path(sys.argv[2] if len(sys.argv) > 2 else "data"))
