"""Scores, the unsure band and bootstrap intervals."""

from __future__ import annotations

import numpy as np

from . import NEG, NEU, POS


def decide(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """negative below lo, positive at or above hi, unsure (neutral) in between."""
    return np.where(q < lo, NEG, np.where(q >= hi, POS, NEU))


def macro_f1(y: np.ndarray, pred: np.ndarray, classes=(NEG, NEU, POS)) -> float:
    f = []
    for c in classes:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        f.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else float("nan")


def binary_accuracy(y: np.ndarray, q: np.ndarray) -> float:
    """Accuracy on positive and negative items only, cut at 0.5 (no unsure band)."""
    keep = y != NEU
    return float(np.mean((q[keep] >= 0.5) == (y[keep] == POS))) if keep.any() else float("nan")


def choose_band(y: np.ndarray, q: np.ndarray, group: np.ndarray | None = None, step: float = 0.01) -> tuple[float, float]:
    """Pick lo <= 0.5 <= hi that maximises 3 way macro F1, averaged over groups (data sets) so no set dominates.

    Uses validation rows only. Ties go to the narrowest band.
    """
    group = np.zeros(len(y), int) if group is None else group
    gs = [g for g in np.unique(group) if (y[group == g] == NEU).any()]  # sets with a neutral class
    best, best_key = (0.5, 0.5), -1.0
    grid = np.round(np.arange(0.0, 0.5 + 1e-9, step), 4)
    for lo in grid:
        for hi in np.round(1 - grid, 4):
            if hi < lo:
                continue
            pred = decide(q, lo, hi)
            score = np.mean([macro_f1(y[group == g], pred[group == g]) for g in gs]) if gs else macro_f1(y, pred)
            key = round(score, 6) - 1e-9 * (hi - lo)
            if key > best_key:
                best, best_key = (float(lo), float(hi)), key
    return best


def bootstrap(stat, n: int, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of stat(idx) over resamples of n items."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.integers(0, n, n)) for _ in range(n_boot)]
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def summarise(y: np.ndarray, q: np.ndarray, lo: float, hi: float, with_ci: bool = True) -> dict:
    pred = decide(q, lo, hi)
    polar = y != NEU
    out = {
        "n": int(len(y)),
        "binary_accuracy": binary_accuracy(y, q),
        "macro_f1_3way": macro_f1(y, pred) if (y == NEU).any() else float("nan"),
        "coverage": float(np.mean(pred[polar] != NEU)) if polar.any() else float("nan"),
        "accuracy_when_sure": float(np.mean(pred[polar & (pred != NEU)] == y[polar & (pred != NEU)])) if (polar & (pred != NEU)).any() else float("nan"),
        "neutral_marked_unsure": float(np.mean(pred[~polar] == NEU)) if (~polar).any() else float("nan"),
    }
    if with_ci:
        out["binary_accuracy_ci"] = bootstrap(lambda i: binary_accuracy(y[i], q[i]), len(y))
        if (y == NEU).any():
            out["macro_f1_3way_ci"] = bootstrap(lambda i: macro_f1(y[i], pred[i]), len(y))
    return out
