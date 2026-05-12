# Movie Review Sentiment Analyzer

Fine-tuned DistilBERT model for binary sentiment classification on movie reviews.
Trained on 25,000 IMDb reviews with 91.2% accuracy.

[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/samboateng190/sentiment-analyzer-app)
[![Model](https://img.shields.io/badge/��%20Model-HuggingFace%20Hub-yellow)](https://huggingface.co/samboateng190/distilbert-imdb-sentiment)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://python.org)

---

## Live Demo

 **[Try it here](https://huggingface.co/spaces/samboateng190/sentiment-analyzer-app)**

---

##  Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| TF-IDF + Logistic Regression (baseline) | 89.46% | 89.46% |
| **DistilBERT (ours)** | **91.19%** | **91.19%** |
| Improvement | +1.73% | +1.73% |

### Training Curves
- Train loss dropped consistently across 3 epochs
- Validation accuracy: 89.8% → 91.0% → 91.2%

---

##  Architecture


---

##  Quick Start

```bash
git clone https://github.com/Boatengs/sentiment-analyzer.git
cd sentiment-analyzer
python3 -m venv venv
source venv/bin/activate
pip install transformers torch gradio
```

Run inference in 5 lines:
```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="samboateng190/distilbert-imdb-sentiment"
)
print(classifier("This movie was absolutely fantastic!"))
```

---

##  Project Structure
---

##  Key Technical Decisions

**Why DistilBERT over BERT?**
DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its performance. On an M3 Mac with shared MPS memory, full BERT caused OOM errors — DistilBERT trained successfully on CPU with comparable accuracy.

**Why max_length=256 instead of 512?**
Our data exploration showed the average review length is 305 tokens, but 86.8% of reviews fit within 256 tokens. Reducing max_length halved memory usage with minimal accuracy impact.

**Why CPU over MPS GPU?**
Apple M3's unified memory architecture allocates a fixed pool (~9GB) shared between CPU and GPU. BERT-sized models exceeded this limit during backpropagation. CPU training was slower but stable.

---

##  Known Limitations

- **Sarcasm:** "Truly groundbreaking cinema" predicted as 99.7% POSITIVE
- **Double negatives:** "I will not advise anyone to watch this" predicted as POSITIVE
- **Short emphatic reviews:** Strong punctuation biases predictions toward POSITIVE
- **Domain:** Trained only on movie reviews — may not generalize to other domains

---

##  Key Learnings

- Fine-tuning a pretrained transformer achieves strong results with minimal data compared to training from scratch
- Hardware constraints drive architectural decisions as much as accuracy targets
- Failure analysis reveals that the model learned emotional intensity rather than true sentiment direction in edge cases

---

##  Training Details

| Parameter | Value |
|-----------|-------|
| Base model | distilbert-base-uncased |
| Dataset | IMDb (25k train, 25k test) |
| Epochs | 3 |
| Batch size | 16 (effective) |
| Learning rate | 2e-5 |
| Max sequence length | 256 |
| Hardware | Apple M3 Mac (CPU) |
| Training time | ~3.6 hours |

---

##  Links

- [Live Demo](https://huggingface.co/spaces/samboateng190/sentiment-analyzer-app)
- [Model on HuggingFace Hub](https://huggingface.co/samboateng190/distilbert-imdb-sentiment)
- [My Portfolio](#)
- [LinkedIn](#)
