"""One off check of raw file formats and labels. Not part of the package."""
import json, collections, traceback
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()

def peek(repo, path, n=3):
    p = hf_hub_download(repo, path, repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    print("==", repo, path, len(rows), "keys:", list(rows[0].keys()))
    for r in rows[:n]: print("   ", {k: (str(v)[:80]) for k, v in r.items()})
    for k in ("label", "label_text", "stars"):
        if k in rows[0]: print("   ", k, dict(sorted(collections.Counter(r[k] for r in rows).items())))

for lang in ["arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"]:
    try: peek("cardiffnlp/tweet_sentiment_multilingual", f"data/{lang}/test.jsonl", 1)
    except Exception: traceback.print_exc()
for lang in ["de", "en", "es", "fr", "ja", "zh"]:
    try: peek("mteb/amazon_reviews_multi", f"{lang}/test.jsonl", 2)
    except Exception: traceback.print_exc()
print(open(hf_hub_download("cardiffnlp/tweet_sentiment_multilingual", "README.md", repo_type="dataset")).read()[:1500])

for repo in ["clapAI/MultiLingualSentiment", "tyqiangz/multilingual-sentiments"]:
    try:
        info = api.dataset_info(repo, files_metadata=True)
        print("==", repo, [(s.rfilename, s.size) for s in info.siblings][:60])
    except Exception: traceback.print_exc()

from datasets import load_dataset
try:
    ds = load_dataset("clapAI/MultiLingualSentiment", split="train")
    print("clapAI train", len(ds), ds.features)
    df = ds.to_pandas()
    for c in df.columns:
        if df[c].nunique() < 60: print("  ", c, df[c].value_counts().head(40).to_dict())
    print(df.head(3).to_string()[:800])
except Exception: traceback.print_exc()
