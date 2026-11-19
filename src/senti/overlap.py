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


def collection_digests(texts) -> set[bytes]:
    out = set()
    for t in texts:
        for k in keys(t):
            out.add(digest(k))
    return out


def seen(df: pd.DataFrame, digests: set[bytes]) -> pd.Series:
    return df["text"].map(lambda t: any(digest(k) in digests for k in keys(t)))


def run(data: Path, out: Path) -> dict:
    from huggingface_hub import HfApi, hf_hub_download

    df = pd.read_parquet(data / "samples.parquet")
    files = [s.rfilename for s in HfApi().dataset_info(COLLECTION).siblings if s.rfilename.endswith(".parquet")]
    result = {"collection": COLLECTION, "by_split": {}}
    for split in ("train", "validation", "test"):
        dig = set()
        for f in [f for f in files if f"/{split}-" in f]:
            dig |= collection_digests(pd.read_parquet(hf_hub_download(COLLECTION, f, repo_type="dataset"), columns=["text"])["text"])
        if not dig:
            continue
        df[f"in_{split}"] = seen(df, dig)
        result["by_split"][split] = int(df[f"in_{split}"].sum())
    cols = [c for c in df.columns if c.startswith("in_")]
    df["in_collection"] = df[cols].any(axis=1)
    table = df.groupby(["set", "lang"])["in_collection"].agg(["sum", "count"]).reset_index()
    result["by_set_lang"] = [{"set": r.set, "lang": r.lang, "seen": int(r["sum"]), "n": int(r["count"])} for _, r in table.iterrows()]
    out.mkdir(parents=True, exist_ok=True)
    df[["id", *cols, "in_collection"]].to_parquet(out / "overlap.parquet", index=False)
    (out / "overlap.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1), flush=True)
    return result


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1] if len(sys.argv) > 1 else "data"), Path(sys.argv[2] if len(sys.argv) > 2 else "data"))
