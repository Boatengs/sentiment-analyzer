"""Turn saved predictions into the results tables, figures and summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics, models

SETS = ["imdb", "amazon23", "amazon23_mt", "tweets", "amazon_multi"]
FRESH = ["amazon23", "amazon23_mt"]
SET_NAMES = {
    "imdb": "IMDb movie reviews (en)",
    "amazon23": "Amazon reviews 2022 to 2023 (en)",
    "amazon23_mt": "Same reviews, machine translated (9 languages)",
    "tweets": "Tweets (8 languages)",
    "amazon_multi": "Amazon reviews (6 languages)",
}
# Test rows a model may have trained on: clapai trained on the whole collection checked in overlap.py;
# lxyuan trained on tyqiangz/multilingual-sentiments, which the collection includes as one source.
SEEN_RULES = {"clapai": "", "lxyuan": "tyqiangz/multilingual-sentiments"}


def seen_ids(o: pd.DataFrame, model: str) -> set[str]:
    if model not in SEEN_RULES:
        return set()
    hit = o["in_collection"] & o["sources"].str.contains(SEEN_RULES[model], regex=False)
    return set(o.loc[hit, "id"])


def load_preds(preds: Path, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name in models.MODELS:
        f = preds / f"{name}.parquet"
        if f.exists():
            p = pd.read_parquet(f)
            m = data.merge(p, on="id", how="inner", validate="one_to_one")
            m["q"] = models.polarity(m[["p_neg", "p_neu", "p_pos"]].to_numpy())
            out[name] = m
    return out


def evaluate(name: str, m: pd.DataFrame, seen: set[str]) -> dict:
    val = m[m["split"] == "val"]
    lo, hi = metrics.choose_band(val["label"].to_numpy(), val["q"].to_numpy(), (val["set"] + "_" + val["lang"]).to_numpy())
    test = m[m["split"] == "test"]
    res = {"band": [lo, hi], "sets": {}, "langs": {}}
    for s in SETS:
        t = test[test["set"] == s]
        if t.empty:
            continue
        r = metrics.summarise(t["label"].to_numpy(), t["q"].to_numpy(), lo, hi)
        clean = t[~t["id"].isin(seen)]
        if len(clean) < len(t):
            r["n_possibly_in_training"] = int(len(t) - len(clean))
            r["binary_accuracy_unseen_only"] = metrics.binary_accuracy(clean["label"].to_numpy(), clean["q"].to_numpy())
        res["sets"][s] = r
        for lang, g in t.groupby("lang"):
            res["langs"][f"{s}/{lang}"] = metrics.summarise(g["label"].to_numpy(), g["q"].to_numpy(), lo, hi, with_ci=False)
    accs = [res["sets"][s]["binary_accuracy"] for s in SETS if s in res["sets"]]
    f1s = [res["sets"][s]["macro_f1_3way"] for s in SETS if s in res["sets"] and not np.isnan(res["sets"][s]["macro_f1_3way"])]
    res["mean_binary_accuracy"] = float(np.mean(accs))
    res["mean_macro_f1_3way"] = float(np.mean(f1s)) if f1s else float("nan")
    # Fresh: reviews written in 2022 and 2023, after every model here was trained, and their translations
    fresh = [s for s in FRESH if s in res["sets"]]
    res["fresh_binary_accuracy"] = float(np.mean([res["sets"][s]["binary_accuracy"] for s in fresh])) if fresh else float("nan")
    res["fresh_macro_f1_3way"] = float(np.mean([res["sets"][s]["macro_f1_3way"] for s in fresh])) if fresh else float("nan")
    en = [res["sets"][s]["binary_accuracy"] for s in ("imdb", "amazon23")] + [res["langs"][k]["binary_accuracy"] for k in ("tweets/en", "amazon_multi/en")]
    res["mean_binary_accuracy_english"] = float(np.mean(en))
    mt = [v["binary_accuracy"] for k, v in res["langs"].items() if k.startswith("amazon23_mt/")]
    res["mean_binary_accuracy_translated"] = float(np.mean(mt)) if mt else float("nan")
    return res


def pct(x: float, d: int = 1) -> str:
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}"


def ci(r: dict, key: str) -> str:
    c = r.get(f"{key}_ci")
    return f"{pct(r[key])} ({pct(c[0])} to {pct(c[1])})" if c else pct(r[key])


def write_markdown(summary: dict, out: Path) -> None:
    res, speed = summary["models"], summary.get("speed", {}).get("models", {})
    order = sorted(res, key=lambda k: -res[k]["fresh_binary_accuracy"])
    L = ["# Results", "", "Every number is computed from the saved predictions in `results/predictions.parquet` by `python -m senti.report`.", ""]
    L += [
        "## Positive or negative: accuracy (%) on the positive and negative test items",
        "",
        "Sorted by the mean of the two fresh sets (Amazon 2022 to 2023, original and translated), the only text no model can have trained on.",
        "",
        "| Model | " + " | ".join(SET_NAMES[s] for s in SETS) + " | Fresh mean |",
        "|---" * (len(SETS) + 2) + "|",
    ]
    for k in order:
        L.append(f"| {k} | " + " | ".join(ci(res[k]["sets"][s], "binary_accuracy") for s in SETS) + f" | {pct(res[k]['fresh_binary_accuracy'])} |")
    L += [
        "",
        "## With an unsure band: 3 way macro F1 (%) against negative, neutral, positive labels",
        "",
        "The band [lo, hi] is chosen per model on validation rows only. IMDb has no neutral items, so it is left out here.",
        "",
        "| Model | Band | "
        + " | ".join(SET_NAMES[x] for x in SETS if x != "imdb")
        + " | Fresh mean | Neutral items marked unsure | Polar items it commits on | Accuracy when it commits |",
        "|---" * (len(SETS) + 5) + "|",
    ]
    for k in sorted(res, key=lambda k: -res[k]["fresh_macro_f1_3way"]):
        r = res[k]
        sets3 = [s for s in SETS if s != "imdb"]
        neu = np.mean([r["sets"][s]["neutral_marked_unsure"] for s in sets3])
        cov = np.mean([r["sets"][s]["coverage"] for s in SETS])
        acc = np.mean([r["sets"][s]["accuracy_when_sure"] for s in SETS])
        L.append(
            f"| {k} | {r['band'][0]:.2f} to {r['band'][1]:.2f} | "
            + " | ".join(ci(r["sets"][s], "macro_f1_3way") for s in sets3)
            + f" | {pct(r['fresh_macro_f1_3way'])} | {pct(neu)} | {pct(cov)} | {pct(acc)} |"
        )
    langs = sorted({k for r in res.values() for k in r["langs"]})
    L += ["", "## Accuracy (%) by language, positive and negative items", "", "| Model | " + " | ".join(langs) + " |", "|---" * (len(langs) + 1) + "|"]
    for k in order:
        L.append(f"| {k} | " + " | ".join(pct(res[k]["langs"][x]["binary_accuracy"]) for x in langs) + " |")
    L += [
        "",
        f"## Speed on one 4 core CPU runner (same {summary.get('speed', {}).get('n_texts', '')} test texts for every model, same machine)",
        "",
        "| Model | Texts per second | Parameters | Licence |",
        "|---|---|---|---|",
    ]
    for k in order:
        s = speed.get(k, {})
        p = s.get("params")
        L.append(f"| {k} | {s.get('texts_per_second', '')} | {'' if p is None else f'{p / 1e6:.0f}M'} | {models.MODELS[k].licence} |")
    unseen = [(k, s) for k in order for s in SETS if "binary_accuracy_unseen_only" in res[k]["sets"].get(s, {})]
    if unseen:
        L += [
            "",
            "## Accuracy (%) with and without test texts the model may have trained on",
            "",
            "clapai: any match in its training collection. lxyuan: matches from tyqiangz/multilingual-sentiments, its training data.",
            "",
            "| Model | Set | Test rows | Possibly seen | All rows | Unseen rows only |",
            "|---|---|---|---|---|---|",
        ]
        for k, s in unseen:
            r = res[k]["sets"][s]
            L.append(f"| {k} | {s} | {r['n']} | {r['n_possibly_in_training']} | {pct(r['binary_accuracy'])} | {pct(r['binary_accuracy_unseen_only'])} |")
    ov = summary.get("overlap")
    if ov:
        L += ["", "## Test texts found word for word in the clapAI training collection", "", "| Set | Language | Seen | Of |", "|---|---|---|---|"]
        L += [f"| {r['set']} | {r['lang']} | {r['seen']} | {r['n']} |" for r in ov["by_set_lang"] if r["seen"]]
        L += ["", "Every other set and language had no matches.", "", "| Source in the collection | Test texts matched |", "|---|---|"]
        L += [f"| {k} | {v} |" for k, v in ov.get("matches_by_source", {}).items()]
    full = summary.get("imdb_full")
    if full:
        n = next(iter(full.values()))["n"]
        L += ["", f"## Full IMDb test set ({n:,} reviews once exact duplicates are removed), cut at 0.5", "", "| Model | Accuracy (%) |", "|---|---|"]
        L += [f"| {k} | {ci(v, 'binary_accuracy')} |" for k, v in full.items()]
    tr = summary.get("distilbert_v2_training")
    if tr:
        L += ["", "## distilbert_v2 training", "", "| Epoch | Validation accuracy (%) | Minutes |", "|---|---|---|"]
        L += [f"| {h['epoch']} | {pct(h['val_accuracy'], 2)} | {h['minutes']} |" for h in tr["history"]]
        L += ["", f"Best epoch on validation: {tr['best_epoch']}."]
    (out / "RESULTS.md").write_text("\n".join(L) + "\n")


def figures(summary: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res, speed = summary["models"], summary.get("speed", {}).get("models", {})
    order = sorted(res, key=lambda k: -res[k]["fresh_binary_accuracy"])
    langs = sorted({k for r in res.values() for k in r["langs"]})
    mat = np.array([[100 * res[k]["langs"][x]["binary_accuracy"] for x in langs] for k in order])
    fig, ax = plt.subplots(figsize=(1.0 + 0.62 * len(langs), 0.9 + 0.42 * len(order)))
    im = ax.imshow(mat, cmap="viridis", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(
        range(len(langs)),
        [x.replace("amazon_multi", "amz").replace("amazon23_mt", "fresh_mt").replace("amazon23", "fresh") for x in langs],
        rotation=60,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks(range(len(order)), order, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] < 80 else "black")
    fig.colorbar(im, ax=ax, label="Accuracy on positive and negative items (%)")
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_language.png", dpi=150)
    plt.close(fig)
    if speed:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        placed: list[tuple[float, float]] = []
        for k in order:
            if k in speed:
                x, y = speed[k]["texts_per_second"], 100 * res[k]["fresh_binary_accuracy"]
                ax.scatter(x, y, color="#3b6fd8")
                # Try above right, below right, then left of the point, and take the first spot no other label is near
                spots = [((4, 3), "left", 0.5), ((4, -11), "left", -1.4), ((-6, -4), "right", -0.5)]

                def free(spot, x=x, y=y):
                    _, ha, yshift = spot
                    lx, ly = np.log10(x) + (0.35 if ha == "left" else -0.35), y + yshift
                    return all(abs(lx - px) > 0.6 or abs(ly - py) > 1.0 for px, py in placed), (lx, ly)

                offset, ha, _ = next((sp for sp in spots if free(sp)[0]), spots[-1])
                lx, ly = free((offset, ha, _))[1]
                ax.annotate(k, (x, y), fontsize=8, xytext=offset, textcoords="offset points", ha=ha)
                placed.append((lx, ly))
        ax.set_xscale("log")
        ax.set_xlabel("Texts per second on a 4 core CPU (log scale)")
        ax.set_ylabel("Accuracy on fresh reviews (%)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "speed_vs_accuracy.png", dpi=150)
        plt.close(fig)


def run(data: Path, preds: Path, out: Path, v2_dir: Path | None = None, speed: Path | None = None) -> dict:
    df = pd.read_parquet(data / "samples.parquet")
    o, overlap = None, None
    if (data / "overlap.parquet").exists():
        o = pd.read_parquet(data / "overlap.parquet")
        overlap = json.loads((data / "overlap.json").read_text())
    p = load_preds(preds, df)
    summary = {
        "models": {k: evaluate(k, m, seen_ids(o, k) if o is not None else set()) for k, m in p.items()},
        "overlap": overlap,
        "counts": df.groupby(["set", "split"]).size().rename("n").reset_index().to_dict("records"),
    }
    if (data / "imdb_full.parquet").exists():
        full = pd.read_parquet(data / "imdb_full.parquet")
        summary["imdb_full"] = {}
        for k in ("tfidf_lr", "distilbert_v1", "distilbert_v2"):
            f = preds / f"{k}.parquet"
            if f.exists():
                m = full.merge(pd.read_parquet(f), on="id")
                if len(m) == len(full):
                    q = models.polarity(m[["p_neg", "p_neu", "p_pos"]].to_numpy())
                    y = m["label"].to_numpy()
                    summary["imdb_full"][k] = {
                        "n": len(m),
                        "binary_accuracy": metrics.binary_accuracy(y, q),
                        "binary_accuracy_ci": metrics.bootstrap(lambda i, y=y, q=q: metrics.binary_accuracy(y[i], q[i]), len(y)),
                    }
    if speed and speed.exists():
        summary["speed"] = json.loads(speed.read_text())
    if v2_dir and (v2_dir / "train.json").exists():
        summary["distilbert_v2_training"] = json.loads((v2_dir / "train.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    rows = [m.loc[m["split"] == "test", ["id", "set", "lang", "label", "source_label", "p_neg", "p_neu", "p_pos"]].assign(model=k) for k, m in p.items()]
    pd.concat(rows, ignore_index=True).to_parquet(out / "predictions.parquet", index=False)
    write_markdown(summary, out)
    figures(summary, out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--preds", type=Path, default=Path("preds"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--v2", type=Path, default=Path("models/distilbert_v2"))
    ap.add_argument("--speed", type=Path, default=Path("preds/speed.json"))
    a = ap.parse_args()
    run(a.data, a.preds, a.out, a.v2, a.speed)


if __name__ == "__main__":
    main()
