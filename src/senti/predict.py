"""Run one model over every sampled text and save its probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data, models


def run(name: str, data_dir: Path, out: Path, local_path: str | None = None, extra: Path | None = None) -> pd.DataFrame:
    df = data.load(data_dir)
    if extra is not None:  # e.g. the full 25,000 review IMDb test set, scored for the IMDb models only
        df = pd.concat([df, pd.read_parquet(extra)], ignore_index=True)
    model = models.load(name, local_path)
    p, seconds = models.timed_predict(model, df["text"].tolist())
    res = pd.DataFrame({"id": df["id"], "p_neg": p[:, 0], "p_neu": p[:, 1], "p_pos": p[:, 2]})
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"{name}.parquet", index=False)
    (out / f"{name}.json").write_text(json.dumps({"model": name, "n": len(df), "seconds": round(seconds, 1)}))
    print(name, len(df), f"{seconds:.0f}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("preds"))
    ap.add_argument("--local-path", default=None)
    ap.add_argument("--extra", type=Path, default=None)
    a = ap.parse_args()
    run(a.model, a.data, a.out, a.local_path, a.extra)


if __name__ == "__main__":
    main()
