import numpy as np
import pandas as pd
import pytest

from senti import NEG, NEU, POS, data, metrics, models, overlap


def test_stars_to_label():
    assert [data.stars_to_label(s) for s in range(1, 6)] == [NEG, NEG, NEU, POS, POS]


def test_sample_is_balanced_disjoint_and_spreads_stars():
    rows = [{"text": f"t{i}", "label": data.stars_to_label(s), "source_label": s} for i, s in enumerate(np.repeat([1, 2, 3, 4, 5], 50))]
    out = data.sample(pd.DataFrame(rows), 5, 20)
    assert out["text"].is_unique
    assert out.groupby(["label", "split"]).size().to_dict() == {(c, sp): n for c in (NEG, NEU, POS) for sp, n in (("test", 20), ("val", 5))}
    neg = out[out["label"] == NEG]["source_label"].value_counts()
    assert abs(neg[1] - neg[2]) <= 1  # 1 and 2 star reviews both in the negative class, evenly


def test_sample_refuses_small_class():
    df = pd.DataFrame({"text": ["a", "b"], "label": [NEG, POS], "source_label": [1, 5]})
    with pytest.raises(ValueError):
        data.sample(df, 1, 1)


@pytest.mark.parametrize(
    "labels,expect",
    [
        ({0: "NEGATIVE", 1: "POSITIVE"}, [[1, 0, 0], [0, 0, 1]]),
        ({0: "positive", 1: "neutral", 2: "negative"}, [[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
        ({0: "1 star", 1: "2 stars", 2: "3 stars", 3: "4 stars", 4: "5 stars"}, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
        ({0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}, [[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]]),
    ],
)
def test_fold_matrix(labels, expect):
    assert models.fold_matrix(labels).tolist() == expect


def test_fold_matrix_rejects_unknown_label():
    with pytest.raises(KeyError):
        models.fold_matrix({0: "LABEL_0"})


def test_polarity_puts_neutral_in_the_middle():
    p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.6, 0.2]])
    assert models.polarity(p).tolist() == [0, 0.5, 1, 0.5]


def test_decide_and_macro_f1():
    q = np.array([0.1, 0.45, 0.5, 0.9])
    assert metrics.decide(q, 0.4, 0.6).tolist() == [NEG, NEU, NEU, POS]
    assert metrics.decide(q, 0.5, 0.5).tolist() == [NEG, NEG, POS, POS]
    y = np.array([NEG, NEU, NEU, POS])
    assert metrics.macro_f1(y, y) == 1.0
    assert metrics.binary_accuracy(y, q) == 1.0


def test_choose_band_finds_the_neutral_gap():
    rng = np.random.default_rng(0)
    y = np.repeat([NEG, NEU, POS], 200)
    q = np.concatenate([rng.uniform(0, 0.3, 200), rng.uniform(0.4, 0.6, 200), rng.uniform(0.7, 1, 200)])
    lo, hi = metrics.choose_band(y, q)
    assert 0.3 <= lo <= 0.4 and 0.6 <= hi <= 0.7
    assert metrics.macro_f1(y, metrics.decide(q, lo, hi)) == 1.0


def test_choose_band_stays_narrow_without_neutral_items():
    y = np.array([NEG, POS] * 50)
    q = np.where(y == POS, 0.9, 0.1)
    assert metrics.choose_band(y, q) == (0.5, 0.5)


def test_summarise_and_bootstrap():
    y = np.array([NEG, POS, NEU, POS] * 25)
    q = np.array([0.1, 0.8, 0.5, 0.3] * 25)
    r = metrics.summarise(y, q, 0.4, 0.6)
    assert r["binary_accuracy"] == pytest.approx(2 / 3)
    assert r["coverage"] == 1.0 and r["neutral_marked_unsure"] == 1.0
    assert r["binary_accuracy_ci"][0] <= r["binary_accuracy"] <= r["binary_accuracy_ci"][1]


def test_overlap_keys_match_body_without_title():
    coll = overlap.collection_digests(["Leider kaputt angekommen, nie wieder bestellt hier."])
    df = pd.DataFrame({"text": ["Schlecht\n\nLeider  kaputt angekommen, nie wieder bestellt hier.", "something else entirely here"]})
    assert overlap.seen(df, coll).tolist() == [True, False]


def test_tfidf_model_outputs_three_columns(toy_texts):
    texts, labels = toy_texts
    p = models.TfidfModel().fit(texts, labels).predict(texts)
    assert p.shape == (len(texts), 3) and np.allclose(p.sum(1), 1) and (p[:, NEU] == 0).all()
    assert np.mean((p[:, POS] >= 0.5) == (labels == 1)) > 0.9


def test_translate_reviews_keeps_title_and_body_apart():
    from senti import translate

    texts = ["Great box\n\nLoved it. Would buy again!", "Meh", "Broken\n\nArrived cracked.\nNo reply from seller."]
    out = translate.translate_reviews(texts, lambda xs: [x.upper() for x in xs])
    assert out == ["GREAT BOX\n\nLOVED IT. WOULD BUY AGAIN!", "MEH", "BROKEN\n\nARRIVED CRACKED. NO REPLY FROM SELLER."]


def test_overlap_records_sources():
    coll = overlap.collection_digests(["first text that is long enough", "second text that is long enough"], [0, 2])
    df = pd.DataFrame({"text": ["first text that is long enough", "second text that is long enough", "unseen"]})
    assert overlap.seen_sources(df, coll).tolist() == [1, 4, 0]


def test_shards_cover_every_row_once(tmp_path):
    from senti import predict

    df = pd.DataFrame({"id": [f"r{i}" for i in range(10)]})
    parts = [predict.shard_rows(df, i, 3) for i in range(3)]
    assert sorted(pd.concat(parts)["id"]) == sorted(df["id"])
    for i, part in enumerate(parts):
        part.assign(p_neg=0.0, p_neu=0.0, p_pos=1.0).to_parquet(tmp_path / f"m.part{i}of3.parquet")
    assert predict.merge_shards(tmp_path) == ["m"]
    assert len(pd.read_parquet(tmp_path / "m.parquet")) == 10
    (tmp_path / "m.part2of3.parquet").unlink()
    with pytest.raises(ValueError):
        predict.merge_shards(tmp_path)
