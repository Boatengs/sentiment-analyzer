"""Run one model over every sampled text and save its probabilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import data, models


def shard_rows(df: pd.DataFrame, shard: int, n_shards: int) -> pd.DataFrame:
    """Every n_shards-th row from position shard, so each shard gets a similar mix of short and long texts."""
    if not 0 <= shard < n_shards:
        raise ValueError("shard must be in [0, n_shards)")
    return df.iloc[shard::n_shards]


def run(name: str, data_dir: Path, out: Path, local_path: str | None = None, extra: Path | None = None, shard: int = 0, n_shards: int = 1) -> pd.DataFrame:
    df = data.load(data_dir)
    if extra is not None:  # e.g. the full 25,000 review IMDb test set, scored for the IMDb models only
        df = pd.concat([df, pd.read_parquet(extra)], ignore_index=True)
    df = shard_rows(df, shard, n_shards)
    suffix = "" if n_shards == 1 else f".part{shard}of{n_shards}"
    model = models.load(name, local_path)
    p, seconds = models.timed_predict(model, df["text"].tolist())
    res = pd.DataFrame({"id": df["id"], "p_neg": p[:, 0], "p_neu": p[:, 1], "p_pos": p[:, 2]})
    out.mkdir(parents=True, exist_ok=True)
    res.to_parquet(out / f"{name}{suffix}.parquet", index=False)
    (out / f"{name}{suffix}.json").write_text(json.dumps({"model": name, "n": len(df), "seconds": round(seconds, 1)}))
    print(name, len(df), f"{seconds:.0f}s", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(models.MODELS))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("preds"))
    ap.add_argument("--local-path", default=None)
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args()
    run(a.model, a.data, a.out, a.local_path, a.extra, a.shard, a.n_shards)


def merge_shards(preds: Path) -> list[str]:
    """Join name.partIofN.parquet files into name.parquet; refuse an incomplete set."""
    done = []
    names = sorted({f.name.split(".part")[0] for f in preds.glob("*.part*of*.parquet")})
    for name in names:
        parts = sorted(preds.glob(f"{name}.part*of*.parquet"))
        n = int(parts[0].name.split("of")[-1].split(".")[0])
        if len(parts) != n:
            raise ValueError(f"{name}: {len(parts)} of {n} shards")
        df = pd.concat([pd.read_parquet(f) for f in parts], ignore_index=True)
        if not df["id"].is_unique:
            raise ValueError(f"{name}: duplicate ids across shards")
        df.to_parquet(preds / f"{name}.parquet", index=False)
        done.append(name)
    return done


if __name__ == "__main__":
    main()
