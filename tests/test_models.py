import numpy as np

from senti import models, train


def test_hub_model_predicts_in_input_order(tiny_model, toy_texts):
    texts, _ = toy_texts
    m = models.HubModel(str(tiny_model), max_length=32, batch_size=7)
    p = m.predict(texts)
    assert p.shape == (len(texts), 3) and np.allclose(p.sum(1), 1)
    one = np.vstack([m.predict([t]) for t in texts[:5]])
    assert np.allclose(one, p[:5], atol=1e-5)  # sorting by length must not shuffle outputs


def test_split_train_is_disjoint():
    tr, va = train.split_train(100, 10, 42)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 100


def test_training_resume_matches_straight_run(tiny_model, toy_texts, tmp_path):
    texts, labels = toy_texts
    recipe = {"epochs": 2, "batch_size": 8, "lr": 1e-3, "val_size": 16, "max_length": 16}
    full = train.run(tmp_path / "full", texts, labels, base=str(tiny_model), recipe=recipe, save_every=3)
    assert train.run(tmp_path / "part", texts, labels, base=str(tiny_model), recipe=recipe, save_every=3, budget_minutes=1e-6) is None
    assert (tmp_path / "part" / "state.pt").exists()
    resumed = None
    for _ in range(20):  # each call trains one save interval, then the budget stops it
        resumed = train.run(tmp_path / "part", texts, labels, base=str(tiny_model), recipe=recipe, save_every=3, budget_minutes=1e-6)
        if resumed:
            break
    if resumed is None:
        resumed = train.run(tmp_path / "part", texts, labels, base=str(tiny_model), recipe=recipe, save_every=3)
    assert [h["val_accuracy"] for h in resumed["history"]] == [h["val_accuracy"] for h in full["history"]]
    assert (tmp_path / "part" / "best" / "config.json").exists() and not (tmp_path / "part" / "state.pt").exists()
    loaded = models.HubModel(str(tmp_path / "part" / "best"), max_length=16)
    assert loaded.predict(texts[:3]).shape == (3, 3)
