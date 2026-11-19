import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WORDS = ["good", "great", "love", "bad", "awful", "hate", "film", "the", "was", "it", "product", "ok"]


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """A randomly initialised two layer DistilBERT with a 12 word vocabulary, saved like a hub model."""
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

    d = tmp_path_factory.mktemp("tiny")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (d / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(d / "vocab.txt")).save_pretrained(d)
    cfg = DistilBertConfig(
        vocab_size=len(vocab),
        dim=16,
        hidden_dim=32,
        n_layers=2,
        n_heads=2,
        max_position_embeddings=64,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    DistilBertForSequenceClassification(cfg).save_pretrained(d)
    return d


@pytest.fixture
def toy_texts():
    import numpy as np

    rng = np.random.default_rng(0)
    pos, neg = ["good", "great", "love"], ["bad", "awful", "hate"]
    texts, labels = [], []
    for i in range(96):
        y = i % 2
        words = list(rng.choice(pos if y else neg, 3)) + list(rng.choice(["the", "film", "was", "it"], 3))
        texts.append(" ".join(rng.permutation(words)))
        labels.append(y)
    return texts, np.array(labels)
