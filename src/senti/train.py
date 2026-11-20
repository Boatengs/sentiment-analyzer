"""Version 2 of the project's own model: version 1's recipe, with the epoch chosen on validation data.

Version 1 fine-tuned distilbert-base-uncased on the 25,000 IMDb training reviews and kept the epoch
with the best accuracy on the IMDb test set. Here 2,500 training reviews are held out as validation,
the model trains on the other 22,500, and the best epoch is picked on those 2,500. The test set is
never touched during training.

Training saves its full state every few hundred steps and stops cleanly before a time budget runs
out, so a later job can continue (GitHub's CPU runners vary a lot in speed).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BASE = "distilbert-base-uncased"
RECIPE = {"epochs": 3, "batch_size": 16, "lr": 2e-5, "weight_decay": 0.01, "max_length": 256, "val_size": 2500, "seed": 42}


def split_train(n: int, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[val_size:]), np.sort(perm[:val_size])


def run(
    out: Path,
    texts: list[str] | None = None,
    labels: np.ndarray | None = None,
    base: str = BASE,
    budget_minutes: float | None = None,
    save_every: int = 200,
    recipe: dict | None = None,
) -> dict | None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    r = {**RECIPE, **(recipe or {})}
    start = time.time()
    torch.manual_seed(r["seed"])
    if texts is None:
        from .models import imdb_train

        tr = imdb_train()
        texts, labels = tr["text"].tolist(), tr["label"].to_numpy()
    tr_idx, va_idx = split_train(len(texts), r["val_size"], r["seed"])
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, id2label={0: "NEGATIVE", 1: "POSITIVE"}, label2id={"NEGATIVE": 0, "POSITIVE": 1}
    )
    no_decay = ["bias", "LayerNorm.weight"]
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": r["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=r["lr"])
    steps_per_epoch = int(np.ceil(len(tr_idx) / r["batch_size"]))
    sched = get_linear_schedule_with_warmup(opt, 0, steps_per_epoch * r["epochs"])

    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.pt"
    history, epoch, step, best = [], 0, 0, -1.0
    if state_path.exists():
        st = torch.load(state_path, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["torch_rng"])
        history, epoch, step, best = st["history"], st["epoch"], st["step"], st["best"]
        print(f"resuming at epoch {epoch} step {step}", flush=True)

    def save_state():
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "history": history,
                "epoch": epoch,
                "step": step,
                "best": best,
            },
            state_path,
        )

    def batches(idx, bs):
        for s in range(0, len(idx), bs):
            b = idx[s : s + bs]
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=r["max_length"], return_tensors="pt")
            yield enc, torch.as_tensor(labels[b], dtype=torch.long)

    step_seconds = []
    while epoch < r["epochs"]:
        order = tr_idx[np.random.default_rng(r["seed"] + epoch).permutation(len(tr_idx))]
        model.train()
        loss_sum, t_epoch = 0.0, time.time()
        bs = r["batch_size"]
        for k, (enc, y) in enumerate(batches(order[step * bs :], bs), start=step):  # skip batches done before a resume
            t = time.time()
            loss = model(**enc, labels=y).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            loss_sum += loss.item()
            step = k + 1
            step_seconds.append(time.time() - t)
            if step % save_every == 0:
                save_state()
                print(
                    f"epoch {epoch + 1} step {step}/{steps_per_epoch} loss {loss_sum / save_every:.4f} {np.mean(step_seconds[-save_every:]):.2f}s/step",
                    flush=True,
                )
                loss_sum = 0.0
            if (
                budget_minutes
                and (time.time() - start) + 60 * 15 + 1.5 * save_every * np.mean(step_seconds[-50:]) > budget_minutes * 60
                and step % save_every == 0
            ):
                print("time budget reached, saving and stopping", flush=True)
                return None
        # validation at the end of each epoch
        model.eval()
        correct = 0
        with torch.inference_mode():
            for enc, y in batches(va_idx, 64):
                correct += int((model(**enc).logits.argmax(-1) == y).sum())
        acc = correct / len(va_idx)
        history.append({"epoch": epoch + 1, "val_accuracy": acc, "minutes": round((time.time() - t_epoch) / 60, 1)})
        print(json.dumps(history[-1]), flush=True)
        if acc > best:
            best = acc
            model.save_pretrained(out / "best")
            tok.save_pretrained(out / "best")
        epoch, step = epoch + 1, 0
        save_state()

    info = {
        "base": base,
        "recipe": r,
        "history": history,
        "best_epoch": int(np.argmax([h["val_accuracy"] for h in history]) + 1),
        "best_val_accuracy": best,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
    }
    (out / "train.json").write_text(json.dumps(info, indent=1))
    state_path.unlink(missing_ok=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("models/distilbert_v2"))
    ap.add_argument("--budget-minutes", type=float, default=None)
    a = ap.parse_args()
    run(a.out, budget_minutes=a.budget_minutes)


if __name__ == "__main__":
    main()
