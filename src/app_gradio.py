import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_path = "samboateng190/distilbert-imdb-sentiment"
tokenizer  = AutoTokenizer.from_pretrained(model_path)
model      = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=-1)

    neg_score = probs[0][0].item()
    pos_score = probs[0][1].item()

    label      = "POSITIVE ✅" if pos_score >= neg_score else "NEGATIVE ❌"
    confidence = max(pos_score, neg_score)
    details    = f"Positive: {pos_score*100:.1f}%  |  Negative: {neg_score*100:.1f}%"

    if confidence < 0.70:
        details += "\n⚠️ Low confidence — review may contain mixed sentiment."

    return label, f"{confidence*100:.1f}%", details

EXAMPLES = [
    ["This movie was absolutely fantastic! The acting was superb and the storyline kept me hooked from start to finish."],
    ["What a complete waste of time. The plot made no sense and the ending was deeply unsatisfying."],
    ["The film has its moments and the lead performance is strong, but the pacing is uneven and the second act drags."],
    ["Oh sure, because what every audience needs is another two hours of predictable plot twists. Truly groundbreaking cinema."],
    ["A masterpiece of foreign cinema. The performances feel raw and authentic. Subtitles are no barrier to its emotional power."],
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=6,
        placeholder="Paste or type a movie review here...",
        label="Movie Review"
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Score Details"),
    ],
    title="🎬 Movie Review Sentiment Analyzer",
    description="Fine-tuned DistilBERT model trained on 25,000 IMDb reviews — 91.2% accuracy",
    examples=EXAMPLES,
)

demo.launch(ssr_mode=False)
