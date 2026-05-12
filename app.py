import gradio as gr
from transformers import pipeline
import time

# Load Model
print("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="samboateng190/distilbert-imdb-sentiment",
    device=-1
)
print("Model loaded!")

def analyze_sentiment(text):
    if not text or text.strip() == "":
        return ""

    start   = time.time()
    result  = classifier(text[:512])[0]
    elapsed = round((time.time() - start) * 1000)

    label      = result["label"]
    score      = result["score"]
    confidence = round(score * 100, 2)
    neg_conf   = round((1 - score) * 100, 2)

    if label == "POSITIVE":
        emoji     = "😊"
        sentiment = "POSITIVE"
        color     = "#10b981"
        pos_conf  = confidence
        neg_c     = neg_conf
    else:
        emoji     = "😞"
        sentiment = "NEGATIVE"
        color     = "#ef4444"
        pos_conf  = neg_conf
        neg_c     = confidence

    return f"""
    <div style="background:var(--bg-card);border:2px solid {color};border-radius:16px;padding:28px;text-align:center;box-shadow:0 4px 20px {color}33;margin:8px 0;">
        <div style="font-size:3.5em;margin-bottom:12px;">{emoji}</div>
        <div style="font-size:1.6em;font-weight:800;color:{color};letter-spacing:2px;margin-bottom:8px;">{sentiment}</div>
        <div style="font-size:2.2em;font-weight:900;color:var(--text-primary);margin-bottom:4px;">{confidence}%</div>
        <div style="color:var(--text-tertiary);font-size:0.8em;margin-bottom:20px;">Confidence Score</div>
        <div style="background:var(--bg-elevated);border-radius:8px;height:10px;overflow:hidden;margin-bottom:20px;">
            <div style="background:linear-gradient(90deg,{color},{color}aa);height:100%;width:{confidence}%;border-radius:8px;"></div>
        </div>
        <div style="background:var(--bg-elevated);border-radius:12px;padding:16px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#10b981;font-weight:600;font-size:0.85em;">😊 Positive</span>
                <span style="color:#10b981;font-weight:700;font-size:0.85em;">{pos_conf}%</span>
            </div>
            <div style="background:var(--bg-primary);border-radius:6px;height:8px;overflow:hidden;margin-bottom:12px;">
                <div style="background:#10b981;height:100%;width:{pos_conf}%;border-radius:6px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                <span style="color:#ef4444;font-weight:600;font-size:0.85em;">😞 Negative</span>
                <span style="color:#ef4444;font-weight:700;font-size:0.85em;">{neg_c}%</span>
            </div>
            <div style="background:var(--bg-primary);border-radius:6px;height:8px;overflow:hidden;">
                <div style="background:#ef4444;height:100%;width:{neg_c}%;border-radius:6px;"></div>
            </div>
        </div>
        <div style="color:var(--text-tertiary);font-size:0.82em;padding-top:12px;border-top:1px solid var(--border);">
            ⚡ Analyzed in {elapsed}ms • DistilBERT • 91.2% Accuracy
        </div>
    </div>
    """

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root {
    --bg-primary:#ffffff;--bg-card:#ffffff;--bg-elevated:#f1f5f9;
    --border:rgba(0,0,0,0.08);--text-primary:#0f172a;--text-secondary:#475569;
    --text-tertiary:#94a3b8;--accent:#6366f1;--accent-dim:rgba(99,102,241,0.1);
    --shadow:0 4px 6px -1px rgba(0,0,0,0.07);--shadow-lg:0 10px 15px -3px rgba(0,0,0,0.08);
    --radius:16px;
}
[data-theme="dark"] {
    --bg-primary:#0f172a;--bg-card:#1e293b;--bg-elevated:#334155;
    --border:rgba(255,255,255,0.08);--text-primary:#f1f5f9;
    --text-secondary:#94a3b8;--text-tertiary:#475569;
    --shadow:0 4px 6px -1px rgba(0,0,0,0.3);--shadow-lg:0 10px 15px -3px rgba(0,0,0,0.4);
}
* { font-family:"Inter",sans-serif !important; box-sizing:border-box; }
body,.gradio-container { background:var(--bg-primary) !important; color:var(--text-primary) !important; transition:all 0.3s ease; }
.gradio-container { max-width:900px !important; margin:0 auto !important; padding:24px !important; }
.app-header { text-align:center; padding:52px 24px 36px; background:linear-gradient(135deg,var(--accent-dim),transparent); border-radius:var(--radius); border:1px solid var(--border); margin-bottom:28px; }
.app-header h1 { font-size:2.6em !important; font-weight:900 !important; background:linear-gradient(135deg,var(--accent),#8b5cf6,#ec4899); -webkit-background-clip:text !important; -webkit-text-fill-color:transparent !important; margin-bottom:14px !important; }
.app-header p { color:var(--text-secondary) !important; font-size:1.05em !important; max-width:560px; margin:0 auto !important; line-height:1.8 !important; }
.badges { display:flex; justify-content:center; gap:10px; flex-wrap:wrap; margin-top:20px; }
.badge { padding:6px 16px; border-radius:20px; font-size:0.75em; font-weight:600; border:1px solid var(--border); background:var(--bg-elevated); color:var(--text-secondary); }
.badge.accent { background:var(--accent-dim); color:var(--accent); border-color:rgba(99,102,241,0.25); }
.badge.positive { background:rgba(16,185,129,0.1); color:#10b981; border-color:rgba(16,185,129,0.25); }
.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:28px; }
.stat-card { background:var(--bg-card); border:1px solid var(--border); border-radius:var(--radius); padding:20px 16px; text-align:center; box-shadow:var(--shadow); transition:all 0.2s ease; }
.stat-card:hover { transform:translateY(-2px); box-shadow:var(--shadow-lg); }
.stat-value { font-size:1.9em; font-weight:900; color:var(--accent); line-height:1; }
.stat-label { font-size:0.7em; color:var(--text-tertiary); font-weight:600; text-transform:uppercase; letter-spacing:0.8px; margin-top:6px; }
.card { background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:var(--radius) !important; padding:28px !important; box-shadow:var(--shadow) !important; }
.theme-toggle { position:fixed !important; top:20px !important; right:20px !important; background:var(--bg-card) !important; border:1px solid var(--border) !important; border-radius:50% !important; width:46px !important; height:46px !important; cursor:pointer !important; font-size:1.2em !important; box-shadow:var(--shadow-lg) !important; transition:all 0.2s !important; z-index:1000 !important; }
.theme-toggle:hover { transform:scale(1.12) rotate(15deg) !important; }
textarea { background:var(--bg-elevated) !important; border:1.5px solid var(--border) !important; border-radius:12px !important; color:var(--text-primary) !important; font-size:0.95em !important; padding:16px !important; transition:border-color 0.2s !important; }
textarea:focus { border-color:var(--accent) !important; box-shadow:0 0 0 3px var(--accent-dim) !important; }
button.primary { background:linear-gradient(135deg,var(--accent),#8b5cf6) !important; border:none !important; border-radius:12px !important; color:white !important; font-weight:700 !important; font-size:1em !important; padding:15px 28px !important; width:100% !important; cursor:pointer !important; transition:all 0.25s !important; box-shadow:0 4px 15px rgba(99,102,241,0.4) !important; }
button.primary:hover { transform:translateY(-2px) !important; box-shadow:0 8px 25px rgba(99,102,241,0.5) !important; }
.example-btn { background:var(--bg-elevated) !important; border:1px solid var(--border) !important; border-radius:10px !important; color:var(--text-secondary) !important; font-size:0.8em !important; padding:8px 14px !important; cursor:pointer !important; transition:all 0.15s !important; text-align:left !important; }
.example-btn:hover { background:var(--accent-dim) !important; border-color:var(--accent) !important; color:var(--accent) !important; }
label { color:var(--text-secondary) !important; font-weight:700 !important; font-size:0.78em !important; text-transform:uppercase !important; letter-spacing:1px !important; }
.footer { text-align:center; padding:28px; color:var(--text-tertiary); font-size:0.78em; border-top:1px solid var(--border); margin-top:36px; line-height:2; }
"""

js = """
function toggleTheme(){
    const root=document.documentElement;
    const btn=document.getElementById("theme-btn");
    const dark=root.getAttribute("data-theme")==="dark";
    root.setAttribute("data-theme",dark?"light":"dark");
    btn.textContent=dark?"🌙":"☀️";
    localStorage.setItem("sentiment-theme",dark?"light":"dark");
}
document.addEventListener("DOMContentLoaded",()=>{
    const saved=localStorage.getItem("sentiment-theme")||"light";
    document.documentElement.setAttribute("data-theme",saved);
    const btn=document.getElementById("theme-btn");
    if(btn)btn.textContent=saved==="dark"?"☀️":"🌙";
});
"""

positive_examples = [
    "This movie was absolutely incredible! The acting was superb and the storyline kept me on the edge of my seat.",
    "I had an amazing experience at this restaurant. The food was delicious and the staff were incredibly friendly.",
    "Best product I have ever bought! It exceeded all my expectations. Highly recommend to everyone!",
]
negative_examples = [
    "Terrible movie. Complete waste of time and money. The plot made no sense and the acting was painfully bad.",
    "Worst customer service I have ever experienced. They were rude, unhelpful and ignored my complaints.",
    "This product broke after just two days. Absolute garbage quality. Do not waste your money on this.",
]

with gr.Blocks(css=css, title="🎬 Sentiment Analyzer") as demo:

    gr.HTML(f"""
    <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()">🌙</button>
    <script>{js}</script>
    <div class="app-header">
        <h1>🎬 Sentiment Analyzer</h1>
        <p>State-of-the-art sentiment analysis powered by fine-tuned DistilBERT.
        Instantly detect whether text is positive or negative with confidence scores.</p>
        <div class="badges">
            <span class="badge accent">DistilBERT</span>
            <span class="badge positive">91.2% Accuracy</span>
            <span class="badge">25K IMDb Reviews</span>
            <span class="badge">Fine-tuned</span>
            <span class="badge">Real-time</span>
        </div>
    </div>
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">91.2%</div><div class="stat-label">Test Accuracy</div></div>
        <div class="stat-card"><div class="stat-value">25K</div><div class="stat-label">Training Reviews</div></div>
        <div class="stat-card"><div class="stat-value">67M</div><div class="stat-label">Parameters</div></div>
        <div class="stat-card"><div class="stat-value">&lt;100ms</div><div class="stat-label">Inference Time</div></div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("### ✍️ Enter Text")
            text_input = gr.Textbox(
                label="Text to Analyze",
                placeholder="Type or paste any text here — movie review, product review, comment...",
                lines=6
            )
            analyze_btn = gr.Button("✨ Analyze Sentiment", variant="primary")
            gr.Markdown("#### 💡 Positive Examples")
            for ex in positive_examples:
                btn = gr.Button(f"📝 {ex[:65]}...", elem_classes=["example-btn"])
                btn.click(fn=lambda x=ex: x, outputs=text_input)
            gr.Markdown("#### 💡 Negative Examples")
            for ex in negative_examples:
                btn = gr.Button(f"📝 {ex[:65]}...", elem_classes=["example-btn"])
                btn.click(fn=lambda x=ex: x, outputs=text_input)

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("### 📊 Analysis Result")
            result_output = gr.HTML(value="""
            <div style="text-align:center;padding:60px 20px;color:var(--text-tertiary);border:2px dashed var(--border);border-radius:16px;margin:8px 0;">
                <div style="font-size:3em;margin-bottom:16px;">🎯</div>
                <div style="font-size:1em;font-weight:600;">Ready to Analyze</div>
                <div style="font-size:0.82em;margin-top:8px;">Enter any text and click<br><b>Analyze Sentiment</b></div>
            </div>
            """)

    gr.HTML("""
    <div style="margin-top:28px;background:var(--bg-card);border:1px solid var(--border);border-radius:16px;padding:28px;box-shadow:var(--shadow);">
        <h3 style="color:var(--text-primary);margin-bottom:20px;font-size:1em;font-weight:700;">🧠 How It Works</h3>
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;">
            <div style="text-align:center;padding:20px;background:var(--bg-elevated);border-radius:12px;">
                <div style="font-size:2em;margin-bottom:10px;">📝</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.9em;margin-bottom:6px;">1. Input Text</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.6;">Enter any text — reviews, comments, tweets</div>
            </div>
            <div style="text-align:center;padding:20px;background:var(--bg-elevated);border-radius:12px;">
                <div style="font-size:2em;margin-bottom:10px;">🔤</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.9em;margin-bottom:6px;">2. Tokenization</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.6;">DistilBERT tokenizes and encodes the text</div>
            </div>
            <div style="text-align:center;padding:20px;background:var(--bg-elevated);border-radius:12px;">
                <div style="font-size:2em;margin-bottom:10px;">🧠</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.9em;margin-bottom:6px;">3. Classification</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.6;">Fine-tuned model classifies with confidence</div>
            </div>
            <div style="text-align:center;padding:20px;background:var(--bg-elevated);border-radius:12px;">
                <div style="font-size:2em;margin-bottom:10px;">📊</div>
                <div style="font-weight:700;color:var(--text-primary);font-size:0.9em;margin-bottom:6px;">4. Results</div>
                <div style="color:var(--text-tertiary);font-size:0.78em;line-height:1.6;">Instant results with confidence bars</div>
            </div>
        </div>
    </div>
    <div class="footer">
        Fine-tuned on 25,000 IMDb Reviews • DistilBERT • 91.2% Test Accuracy • Beats TF-IDF by +1.73% •
        <a href="https://github.com/Boatengs/sentiment-analyzer" target="_blank" style="color:var(--accent);">GitHub</a> •
        <a href="https://huggingface.co/samboateng190/distilbert-imdb-sentiment" target="_blank" style="color:var(--accent);">Model</a>
    </div>
    """)

    analyze_btn.click(fn=analyze_sentiment, inputs=text_input, outputs=result_output)

demo.launch()
