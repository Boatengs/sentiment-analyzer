"""Build the validation and test samples.

Every row has: id, set, lang, domain, text, label (0 negative, 1 neutral, 2 positive),
source_label (the original star rating or class) and split (val or test).
Validation rows are only used to choose each model's unsure band; test rows only to score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import NEG, NEU, POS, SEED

AMAZON23 = "McAuley-Lab/Amazon-Reviews-2023"
AMAZON23_CATEGORIES = ["Subscription_Boxes", "Magazine_Subscriptions", "Gift_Cards", "Digital_Music", "Health_and_Personal_Care"]
AMAZON23_FROM = "2022-01-01"  # after the release of every model that publishes a date before 2022
TWEET_LANGS = {"arabic": "ar", "english": "en", "french": "fr", "german": "de", "hindi": "hi", "italian": "it", "portuguese": "pt", "spanish": "es"}
AMAZON_MULTI_LANGS = ["de", "en", "es", "fr", "ja", "zh"]

# per class counts (val, test)
PER_CLASS = {"imdb": (100, 300), "amazon23": (100, 300), "tweets": (70, 220), "amazon_multi": (100, 300)}


def stars_to_label(stars: int, top: int = 5) -> int:
    """1-2 stars negative, 3 neutral, 4-5 positive (on a 1 to 5 scale)."""
    if top != 5:
        raise ValueError("only 5 star scales")
    return NEG if stars <= 2 else NEU if stars == 3 else POS


def sample(df: pd.DataFrame, n_val: int, n_test: int, seed: int = SEED, by: str = "label") -> pd.DataFrame:
    """Take n_val + n_test rows per class without overlap; within a class, spread evenly over source_label."""
    rng = np.random.default_rng(seed)
    out = []
    for _, g in df.groupby(by, sort=True):
        need = n_val + n_test
        if len(g) < need:
            raise ValueError(f"class {g[by].iloc[0]} has {len(g)} rows, need {need}")
        # round robin over source labels so 1 and 2 star reviews are both in the negative class
        parts = [s.sample(frac=1, random_state=int(rng.integers(1 << 31))) for _, s in g.groupby("source_label", sort=True)]
        order = pd.concat([p.assign(_r=np.arange(len(p))) for p in parts]).sort_values(["_r", "source_label"], kind="stable")
        take = order.head(need).drop(columns="_r").sample(frac=1, random_state=int(rng.integers(1 << 31)))
        take = take.assign(split=["val"] * n_val + ["test"] * n_test)
        out.append(take)
    return pd.concat(out, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["text"].astype(str).str.strip().str.len() > 0]
    return df.drop_duplicates("text")


def _hub(repo: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, path, repo_type="dataset"))


def load_imdb() -> pd.DataFrame:
    df = pd.read_parquet(_hub("stanfordnlp/imdb", "plain_text/test-00000-of-00001.parquet"))
    df = df.assign(source_label=df["label"], label=np.where(df["label"] == 1, POS, NEG))
    df["id"] = "imdb_test_" + df.index.astype(str)
    return clean(df.assign(lang="en", domain="movies", set="imdb"))


def load_amazon23() -> pd.DataFrame:
    frames = []
    for cat in AMAZON23_CATEGORIES:
        d = pd.read_json(_hub(AMAZON23, f"raw/review_categories/{cat}.jsonl"), lines=True)
        d = d[pd.to_datetime(d["timestamp"], unit="ms") >= AMAZON23_FROM]
        d = d.assign(category=cat, text=(d["title"].fillna("").str.strip() + "\n\n" + d["text"].fillna("").str.strip()).str.strip())
        d["id"] = "amazon23_" + d["user_id"].astype(str) + "_" + d["asin"].astype(str) + "_" + d["timestamp"].astype(str)
        frames.append(d[["id", "text", "rating", "category", "timestamp"]])
    df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    df = df.assign(source_label=df["rating"].astype(int), label=df["rating"].astype(int).map(stars_to_label))
    return clean(df.assign(lang="en", domain="products", set="amazon23"))


def load_tweets() -> pd.DataFrame:
    frames = []
    for name, code in TWEET_LANGS.items():
        rows = [json.loads(line) for line in open(_hub("cardiffnlp/tweet_sentiment_multilingual", f"data/{name}/test.jsonl"), encoding="utf-8")]
        d = pd.DataFrame(rows).assign(lang=code)
        d["id"] = f"tweets_{code}_" + d.index.astype(str)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    lab = df["label"].astype(int)  # 0 negative, 1 neutral, 2 positive (dataset card)
    return clean(df.assign(source_label=lab, label=lab, domain="tweets", set="tweets"))


def load_amazon_multi() -> pd.DataFrame:
    frames = []
    for code in AMAZON_MULTI_LANGS:
        rows = [json.loads(line) for line in open(_hub("mteb/amazon_reviews_multi", f"{code}/test.jsonl"), encoding="utf-8")]
        frames.append(pd.DataFrame(rows).assign(lang=code))
    df = pd.concat(frames, ignore_index=True)
    stars = df["label"].astype(int) + 1  # stored as 0 to 4
    df = df.assign(source_label=stars, label=stars.map(stars_to_label))
    return clean(df.assign(domain="products", set="amazon_multi"))


COLUMNS = ["id", "set", "lang", "domain", "text", "label", "source_label", "split"]


def build(out: Path) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    parts = []
    imdb = load_imdb()
    parts.append(sample(imdb, *PER_CLASS["imdb"]))
    am = load_amazon23()
    parts.append(sample(am, *PER_CLASS["amazon23"]).assign(set="amazon23"))
    tw = load_tweets()
    parts += [sample(g, *PER_CLASS["tweets"]) for _, g in tw.groupby("lang")]
    mu = load_amazon_multi()
    parts += [sample(g, *PER_CLASS["amazon_multi"]) for _, g in mu.groupby("lang")]
    df = pd.concat(parts, ignore_index=True)
    extra = [c for c in ("category",) if c in df.columns]
    df = df[COLUMNS + extra]
    assert df["id"].is_unique and df["text"].is_unique
    df.to_parquet(out / "samples.parquet", index=False)
    stats = {
        "amazon23_pool": {"reviews_2022_on": len(am), "by_rating": am["source_label"].value_counts().sort_index().to_dict()},
        "rows": df.groupby(["set", "lang", "split", "label"]).size().reset_index(name="n").to_dict("records"),
    }
    (out / "data_stats.json").write_text(json.dumps(stats, indent=1, default=int))
    print(df.groupby(["set", "split"]).size(), flush=True)
    return df


def load(data: Path) -> pd.DataFrame:
    return pd.read_parquet(data / "samples.parquet")


def build_imdb_full(out: Path) -> pd.DataFrame:
    """All 25,000 IMDb test reviews, to recheck the version 1 accuracy claim on the same set."""
    df = load_imdb().reset_index(drop=True)
    df = df.assign(id="imdbfull_" + df.index.astype(str), set="imdb_full", split="full")[COLUMNS]
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "imdb_full.parquet", index=False)
    return df


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    build(target)
    build_imdb_full(target)
