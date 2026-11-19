"""Time every model on the same 256 texts on the same machine, so speeds are comparable."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import SEED, models


def run(data: Path, out: Path, local: dict[str, str], n: int = 256, repeats: int = 2) -> dict:
    df = pd.read_parquet(data / "samples.parquet")
    df = df[df["split"] == "test"].groupby("set", group_keys=False).apply(lambda g: g.sample(n // 4, random_state=SEED))
    texts = df["text"].tolist()
    res = {"n_texts": len(texts), "mean_chars": float(df["text"].str.len().mean()), "models": {}}
    for name in models.MODELS:
        m = models.load(name, local.get(name))
        m.predict(texts[:16])  # warm up
        best = min(models.timed_predict(m, texts)[1] for _ in range(repeats))
        params = sum(p.numel() for p in m.model.parameters()) if hasattr(m, "model") else None
        res["models"][name] = {"texts_per_second": round(len(texts) / best, 1), "params": params}
        print(name, res["models"][name], flush=True)
        del m
    out.mkdir(parents=True, exist_ok=True)
    (out / "speed.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys

    run(Path(sys.argv[1]), Path(sys.argv[2]), {"distilbert_v2": sys.argv[3]} if len(sys.argv) > 3 else {})
