"""Gradio demo: positive, negative or unsure, with the model and unsure band chosen in the version 2 benchmark.

The bands come from results/summary.json. They were picked on validation rows only, to maximise
3 way macro F1 (negative, neutral, positive). A score inside the band is reported as unsure.
"""

import gradio as gr
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CHOICES = {
    "Reviews, any of the tested languages (nlptown, MIT licence)": ("nlptown/bert-base-multilingual-uncased-sentiment", (0.24, 0.74)),
    "Social media posts (Cardiff NLP XLM-T, no licence listed)": ("cardiffnlp/twitter-xlm-roberta-base-sentiment", (0.22, 0.67)),
    "Highest accuracy, slow on CPU (clapAI ModernBERT, Apache 2.0)": ("clapAI/modernBERT-base-multilingual-sentiment", (0.23, 0.76)),
}

TO_CLASS = {
    "negative": 0, "neutral": 1, "positive": 2,
    "1 star": 0, "2 stars": 0, "3 stars": 1, "4 stars": 2, "5 stars": 2,
    "very negative": 0, "very positive": 2,
}  # fmt: skip

_cache = {}


def load(hub_id):
    if hub_id not in _cache:
        tok = AutoTokenizer.from_pretrained(hub_id)
        model = AutoModelForSequenceClassification.from_pretrained(hub_id).eval()
        fold = np.zeros((len(model.config.id2label), 3))
        for i, name in model.config.id2label.items():
            fold[int(i), TO_CLASS[str(name).strip().lower()]] = 1.0
        _cache[hub_id] = (tok, model, fold)
    return _cache[hub_id]


def analyse(text, choice):
    if not text or not text.strip():
        return "Type or paste some text.", {}
    hub_id, (lo, hi) = CHOICES[choice]
    tok, model, fold = load(hub_id)
    limit = min(512, getattr(model.config, "max_position_embeddings", 512))
    limit = limit - 2 if limit == 514 else limit
    with torch.inference_mode():
        enc = tok([text], truncation=True, max_length=limit, return_tensors="pt")
        p = torch.softmax(model(**enc).logits.float(), dim=-1).numpy()[0] @ fold
    score = float(p[2] + 0.5 * p[1])
    if score < lo:
        verdict = "Negative"
    elif score > hi:
        verdict = "Positive"
    else:
        verdict = "Unsure (mixed or neutral)"
    detail = f"{verdict}\n\nScore {score:.2f} on a 0 (negative) to 1 (positive) scale. This model reports unsure between {lo:.2f} and {hi:.2f}."
    return detail, {"negative": float(p[0]), "neutral": float(p[1]), "positive": float(p[2])}


with gr.Blocks(title="Sentiment analyzer") as demo:
    gr.Markdown(
        "# Sentiment analyzer\n"
        "Positive, negative or unsure. The models and unsure bands come from a benchmark of 11 models on reviews written in 2022 and 2023, "
        "their machine translations into 9 languages, and tweets in 8 languages. Details: "
        "[github.com/Boatengs/sentiment-analyzer](https://github.com/Boatengs/sentiment-analyzer)"
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text", lines=6)
            choice = gr.Radio(list(CHOICES), value=next(iter(CHOICES)), label="Model")
            go = gr.Button("Analyse", variant="primary")
        with gr.Column():
            verdict = gr.Textbox(label="Result", lines=4)
            probs = gr.Label(label="Probabilities")
    gr.Examples(
        [
            ["The box arrived crushed and two of the items were missing.", next(iter(CHOICES))],
            ["Das Produkt ist in Ordnung, aber nichts Besonderes.", next(iter(CHOICES))],
            ["Me encanta, lo uso todos los días.", next(iter(CHOICES))],
        ],
        [text, choice],
    )
    go.click(analyse, [text, choice], [verdict, probs])

if __name__ == "__main__":
    demo.launch()
