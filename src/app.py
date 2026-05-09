import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import plotly.graph_objects as go

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ── Load Model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "../models/distilbert-sentiment-final"
    tokenizer  = AutoTokenizer.from_pretrained(model_path)
    model      = AutoModelForSequenceClassification.from_pretrained(model_path)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        return_all_scores=True
    )

sentiment_pipeline = load_model()

# ── Example Reviews ──────────────────────────────────────────────
EXAMPLES = [
    {
        "label": "⭐ Clear Positive",
        "text": "This movie was absolutely fantastic! The acting was superb, the storyline kept me hooked from start to finish, and the cinematography was breathtaking. One of the best films I have seen in years!"
    },
    {
        "label": "💔 Clear Negative",
        "text": "What a complete waste of time. The plot made no sense, the acting was wooden, and the ending was deeply unsatisfying. I cannot believe I sat through the entire thing. Avoid at all costs."
    },
    {
        "label": "🤔 Ambiguous (Mixed)",
        "text": "The film has its moments and the lead performance is strong, but the pacing is uneven and the second act drags considerably. Worth watching once but unlikely to revisit."
    },
    {
        "label": "😏 Sarcastic Negative",
        "text": "Oh sure, because what every audience needs is another two hours of predictable plot twists and cardboard characters. Truly groundbreaking cinema. The director must be very proud."
    },
    {
        "label": "🌍 Foreign Film Positive",
        "text": "A masterpiece of foreign cinema. The director crafts each scene with precision and the performances feel raw and authentic. Subtitles are no barrier to the emotional power of this film."
    },
]

# ── Helper Functions ─────────────────────────────────────────────
def predict(text):
    raw = sentiment_pipeline(text, truncation=True, max_length=256)

    # Flatten output if nested
    results = raw[0] if isinstance(raw[0], list) else raw

    # Build a score dict — handle any label naming convention
    scores = {r["label"]: r["score"] for r in results}

    # Detect label format and extract pos/neg scores
    if "POSITIVE" in scores and "NEGATIVE" in scores:
        pos_score = scores["POSITIVE"]
        neg_score = scores["NEGATIVE"]
    elif "LABEL_1" in scores and "LABEL_0" in scores:
        pos_score = scores["LABEL_1"]  # LABEL_1 = POSITIVE
        neg_score = scores["LABEL_0"]  # LABEL_0 = NEGATIVE
    else:
        # Last resort — sort by score, highest = positive
        sorted_scores = sorted(results, key=lambda x: x["score"], reverse=True)
        pos_score     = sorted_scores[0]["score"]
        neg_score     = sorted_scores[1]["score"] if len(sorted_scores) > 1 else 1 - pos_score

    label      = "POSITIVE" if pos_score >= neg_score else "NEGATIVE"
    confidence = max(pos_score, neg_score)
    return label, confidence, neg_score, pos_score


def confidence_chart(neg_score, pos_score):
    fig = go.Figure(go.Bar(
        x=[neg_score * 100, pos_score * 100],
        y=["NEGATIVE", "POSITIVE"],
        orientation="h",
        marker_color=["#E74C3C", "#2ECC71"],
        text=[f"{neg_score*100:.1f}%", f"{pos_score*100:.1f}%"],
        textposition="inside",
        textfont=dict(size=14, color="white")
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="Confidence (%)"),
        yaxis=dict(title=""),
        height=180,
        margin=dict(l=10, r=10, t=10, b=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def show_results(text):
    with st.spinner("Analyzing..."):
        label, confidence, neg_score, pos_score = predict(text)

    st.divider()
    st.markdown("#### 📊 Results")

    if label == "POSITIVE":
        st.success(f"✅ POSITIVE  —  {confidence*100:.1f}% confidence")
    else:
        st.error(f"❌ NEGATIVE  —  {confidence*100:.1f}% confidence")

    st.plotly_chart(confidence_chart(neg_score, pos_score), use_container_width=True)

    word_count = len(text.split())
    st.caption(f"📝 {word_count} words analyzed")

    if confidence < 0.70:
        st.info("💡 Low confidence — this review likely contains mixed or ambiguous sentiment.")

    if word_count > 350:
        st.caption("⚠️ Review exceeds 256 tokens — text was truncated for analysis.")


# ── Main UI ──────────────────────────────────────────────────────
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Fine-tuned **DistilBERT** model trained on 25,000 IMDb reviews — 91.2% accuracy")
st.divider()

# ── Example Buttons ──────────────────────────────────────────────
st.markdown("#### 💡 Try an example")
cols = st.columns(len(EXAMPLES))
for i, (col, example) in enumerate(zip(cols, EXAMPLES)):
    if col.button(example["label"], key=f"ex_{i}", use_container_width=True):
        st.session_state["input_text"]  = example["text"]
        st.session_state["auto_analyze"] = True
        st.rerun()

st.divider()

# ── Text Input ───────────────────────────────────────────────────
st.markdown("#### ✍️ Or write your own review")
text_input = st.text_area(
    label="",
    value=st.session_state.get("input_text", ""),
    placeholder="Paste or type a movie review here...",
    height=180,
    key="text_area"
)

# ── Analyze Button ───────────────────────────────────────────────
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter a review first!")
    else:
        show_results(text_input)

# ── Auto Analyze on Example Click ────────────────────────────────
if st.session_state.get("auto_analyze"):
    st.session_state["auto_analyze"] = False
    auto_text = st.session_state.get("input_text", "")
    if auto_text.strip():
        show_results(auto_text)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 About This Model")
    st.markdown("""
    **Model:** DistilBERT (base, uncased)
    
    **Dataset:** IMDb — 25,000 training reviews
    
    **Training:** 3 epochs on CPU (M3 Mac)
    
    **Results:**
    - ✅ Accuracy: **91.2%**
    - ✅ F1 Score: **91.2%**
    - ✅ Beat TF-IDF baseline by **+1.73%**
    """)
    st.divider()
    st.markdown("## ⚠️ Known Limitations")
    st.markdown("""
    - Struggles with **sarcasm**
    - Mixed reviews confuse the model
    - Truncates reviews over ~350 words
    - Trained only on movie reviews
    """)
    st.divider()
    st.markdown("## 🔗 Links")
    st.markdown("""
    - [GitHub Repo](#)
    - [HuggingFace Model](#)
    - [My Portfolio](#)
    """)