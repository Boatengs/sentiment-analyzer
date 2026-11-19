"""One off check that every candidate model and dataset loads. Not part of the package."""
import json, time, traceback
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

MODELS = [
    "samurvivor-07/distilbert-imdb-sentiment",
    "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    "siebert/sentiment-roberta-large-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment",
    "tabularisai/multilingual-sentiment-analysis",
    "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    "clapAI/modernBERT-base-multilingual-sentiment",
]
DATASETS = [
    "stanfordnlp/imdb", "McAuley-Lab/Amazon-Reviews-2023", "cardiffnlp/tweet_eval",
    "cardiffnlp/tweet_sentiment_multilingual", "mteb/amazon_reviews_multi", "Yelp/yelp_review_full",
    "mteb/tweet_sentiment_extraction", "SetFit/amazon_reviews_multi_en",
]
TEXTS = ["The battery died after two days and support never answered.", "Absolutely love it, works perfectly.",
         "Das Produkt ist leider kaputt angekommen.", "Me encanta, funciona de maravilla."]

from transformers import pipeline
for m in MODELS:
    print("=" * 20, m, flush=True)
    try:
        info = api.model_info(m, files_metadata=False)
        card = info.card_data.to_dict() if info.card_data else {}
        print("license:", card.get("license"), "| datasets:", card.get("datasets"), "| langs:", str(card.get("language"))[:120],
              "| created:", info.created_at, "| downloads:", info.downloads)
        t = time.time(); clf = pipeline("text-classification", model=m, top_k=None, truncation=True)
        print("load s:", round(time.time() - t, 1), "| labels:", clf.model.config.id2label,
              "| params:", sum(p.numel() for p in clf.model.parameters()))
        t = time.time(); out = clf(TEXTS)
        print("4 texts s:", round(time.time() - t, 2))
        for x, o in zip(TEXTS, out):
            print("  ", x[:40], {d["label"]: round(d["score"], 3) for d in o})
    except Exception:
        traceback.print_exc()

for d in DATASETS:
    print("=" * 20, d, flush=True)
    try:
        info = api.dataset_info(d)
        card = info.card_data.to_dict() if info.card_data else {}
        print("license:", card.get("license"), "| configs:", [c.get("config_name") for c in (card.get("configs") or [])][:30])
        files = [s.rfilename for s in info.siblings]
        print("n files:", len(files), files[:40])
    except Exception:
        traceback.print_exc()

# Amazon 2023: sizes of review files, and timestamps in one small category
try:
    files = api.list_repo_tree("McAuley-Lab/Amazon-Reviews-2023", path_in_repo="raw/review_categories", repo_type="dataset")
    files = sorted([(f.size, f.path) for f in files if hasattr(f, "size")])
    print("smallest review files:", files[:12])
    print("largest:", files[-3:])
    p = hf_hub_download("McAuley-Lab/Amazon-Reviews-2023", files[0][1], repo_type="dataset")
    import pandas as pd
    df = pd.read_json(p, lines=True)
    print(df.columns.tolist(), len(df))
    ts = pd.to_datetime(df["timestamp"], unit="ms")
    print("years:", ts.dt.year.value_counts().sort_index().to_dict())
    print("ratings:", df["rating"].value_counts().sort_index().to_dict())
    print(df[["rating", "title", "text"]].head(3).to_string()[:600])
except Exception:
    traceback.print_exc()

from datasets import load_dataset
for name, cfg, split in [("cardiffnlp/tweet_sentiment_multilingual", "german", "test"), ("mteb/amazon_reviews_multi", "de", "test"),
                         ("cardiffnlp/tweet_eval", "sentiment", "test"), ("mteb/tweet_sentiment_extraction", None, "test")]:
    try:
        ds = load_dataset(name, cfg, split=split)
        print("==", name, cfg, split, len(ds), ds.features, ds[0])
    except Exception:
        traceback.print_exc()
