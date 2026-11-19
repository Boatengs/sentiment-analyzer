"""Every model is turned into the same output: probabilities for negative, neutral, positive.

Two class models put 0 on neutral. Five class models (stars, or very negative to very positive)
fold 1 and 2 into negative, 3 into neutral, 4 and 5 into positive.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import NEG, NEU, POS


@dataclass(frozen=True)
class Spec:
    name: str
    hub_id: str
    languages: str  # what the model card says it covers
    licence: str  # from the model card metadata
    note: str = ""
    max_length: int = 512


MODELS = {
    s.name: s
    for s in [
        Spec("tfidf_lr", "", "en", "ours", "TF-IDF + logistic regression trained on the 25,000 IMDb training reviews"),
        Spec(
            "distilbert_v1", "samurvivor-07/distilbert-imdb-sentiment", "en", "none listed", "Version 1 of this project; epoch chosen on the IMDb test set", 256
        ),
        Spec("distilbert_v2", "", "en", "ours", "Same recipe as version 1, epoch chosen on a validation split taken from IMDb train", 256),
        Spec("distilbert_sst2", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", "en", "apache-2.0", "Default Hugging Face sentiment pipeline"),
        Spec("siebert", "siebert/sentiment-roberta-large-english", "en", "none listed", "RoBERTa large fine-tuned on 15 English datasets"),
        Spec("cardiff_en", "cardiffnlp/twitter-roberta-base-sentiment-latest", "en", "cc-by-4.0", "RoBERTa base trained on 124M tweets, tuned on TweetEval"),
        Spec(
            "cardiff_xlmt", "cardiffnlp/twitter-xlm-roberta-base-sentiment", "multilingual", "none listed", "XLM-RoBERTa base trained on tweets in 8 languages"
        ),
        Spec(
            "nlptown",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "en nl de fr it es",
            "mit",
            "Multilingual BERT predicting 1 to 5 stars for product reviews",
        ),
        Spec(
            "lxyuan",
            "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
            "12 languages",
            "apache-2.0",
            "Multilingual DistilBERT distilled from a zero shot teacher",
        ),
        Spec("clapai", "clapAI/modernBERT-base-multilingual-sentiment", "16 languages", "apache-2.0", "Multilingual ModernBERT trained on 3.1M labelled texts"),
        Spec(
            "tabularisai",
            "tabularisai/multilingual-sentiment-analysis",
            "22 languages",
            "cc-by-nc-4.0",
            "Multilingual DistilBERT trained on synthetic data; non commercial licence",
        ),
    ]
}

_NAME_TO_CLASS = {
    "negative": NEG,
    "neutral": NEU,
    "positive": POS,
    "1 star": NEG,
    "2 stars": NEG,
    "3 stars": NEU,
    "4 stars": POS,
    "5 stars": POS,
    "very negative": NEG,
    "very positive": POS,
}


def fold_matrix(id2label: dict) -> np.ndarray:
    """(n_model_classes, 3) matrix that sums model class probabilities into negative, neutral, positive."""
    m = np.zeros((len(id2label), 3))
    for i, name in id2label.items():
        key = str(name).strip().lower()
        if key not in _NAME_TO_CLASS:
            raise KeyError(f"unknown label {name!r}")
        m[int(i), _NAME_TO_CLASS[key]] = 1.0
    return m


def polarity(p3: np.ndarray) -> np.ndarray:
    """One number from 0 (surely negative) to 1 (surely positive); neutral mass sits in the middle."""
    return p3[:, POS] + 0.5 * p3[:, NEU]


class HubModel:
    def __init__(self, path: str, max_length: int = 512, batch_size: int = 16):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.fold = fold_matrix(self.model.config.id2label)
        limit = getattr(self.model.config, "max_position_embeddings", 512)
        # RoBERTa style models reserve 2 positions for padding offsets
        self.max_length = min(max_length, limit - 2 if limit in (514, 130) else limit)
        self.batch_size = batch_size

    def predict(self, texts: list[str]) -> np.ndarray:
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch: much faster on CPU
        out = np.zeros((len(texts), 3))
        with self.torch.inference_mode():
            for s in range(0, len(texts), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self.tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
                p = self.torch.softmax(self.model(**enc).logits.float(), dim=-1).numpy()
                out[idx] = p @ self.fold
        return out


class TfidfModel:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self.vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
        self.clf = LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, texts, labels) -> TfidfModel:
        self.clf.fit(self.vec.fit_transform(texts), labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        p = self.clf.predict_proba(self.vec.transform(texts))  # columns follow classes_ = [0, 1]
        out = np.zeros((len(texts), 3))
        out[:, NEG], out[:, POS] = p[:, 0], p[:, 1]
        return out


def imdb_train():
    import pandas as pd
    from huggingface_hub import hf_hub_download

    return pd.read_parquet(hf_hub_download("stanfordnlp/imdb", "plain_text/train-00000-of-00001.parquet", repo_type="dataset"))


def load(name: str, local_path: str | None = None):
    spec = MODELS[name]
    if name == "tfidf_lr":
        tr = imdb_train()
        return TfidfModel().fit(tr["text"].tolist(), tr["label"].to_numpy())
    return HubModel(local_path or spec.hub_id, max_length=spec.max_length)


def timed_predict(model, texts: list[str]) -> tuple[np.ndarray, float]:
    t = time.perf_counter()
    p = model.predict(texts)
    return p, time.perf_counter() - t
