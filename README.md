# 📩 AT&T SMS Spam Detector — NLP & Deep Learning

> Automatic spam detection for SMS messages, comparing a classic ML baseline, a custom neural network and a fine-tuned **DistilBERT** transformer.
> *Jedha Data Science certification project — RNCP35288, Block 4 (Deep Learning on unstructured data).*

---

## 🎯 Business problem
AT&T users receive unwanted spam SMS. The goal is to **flag spam automatically from the message text alone**, while:
- catching as many spams as possible (**recall**),
- without blocking legitimate messages (**precision**).

**Main metric: F1-score** (balance between both), completed by a confusion-matrix error analysis.

## 📊 Data
- **SMS Spam Collection v1**: 5,572 messages labelled *ham* (legitimate) or *spam*
- 403 duplicates removed → **5,169 messages**
- Imbalanced classes: ~13 % spam
- Insight: spam messages are about **twice as long** as legitimate ones (~138 vs ~70 characters on average)

## 🛠️ Approach
1. **Cleaning**: lowercasing, punctuation removal, whitespace normalisation, label encoding
2. **Stratified 80/20 split** (test set: 1,034 messages)
3. **Three models of increasing complexity**:

| # | Model | Details |
|---|---|---|
| 1 | **Logistic Regression** (baseline) | TF-IDF (1-2 grams, 5,000 features), `class_weight="balanced"` |
| 2 | **Custom neural network** | Keras: Embedding (dim 128) → GlobalAveragePooling → Dense 64 → Dense 32 → sigmoid · ~987k params · 10 epochs |
| 3 | **DistilBERT** (transfer learning) | `distilbert-base-uncased` fine-tuned · max length 128 · Adam lr 2e-5 · 2 epochs · batch 16 |

## ✅ Results (test set, 1,034 messages)

| Model | Accuracy | Precision | Recall | **F1-score** | Spams missed | Ham wrongly blocked |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.977 | 0.928 | 0.885 | 0.906 | 15 | 9 |
| Custom Embedding NN | 0.984 | **0.983** | 0.885 | 0.932 | 15 | **2** |
| **DistilBERT** | **0.987** | 0.947 | **0.954** | **0.951** | **6** | 7 |

**➡️ Recommended model: DistilBERT** — best F1 (+4.9 points vs baseline) and it misses **2.5× fewer spams**.
If wrongly blocking a legitimate SMS is considered more costly than letting a spam through, the lighter **custom embedding model** is a strong alternative (only 2 false positives, much faster at inference).

## ⚠️ Limitations
- Small, dated dataset (2012), English only: modern spam is more sophisticated
- DistilBERT is heavier and slower than the custom model

## 🚀 Next steps towards production
- Serve the model through a **FastAPI** REST API in a **Docker** container
- Monitor data drift and retrain regularly; active learning on low-confidence messages
- Speed-up options: quantization, TinyBERT; consider multilingual models

## 🧰 Tech stack
`Python` · `pandas` · `scikit-learn` · `TensorFlow / Keras` · `Hugging Face Transformers` · `matplotlib / seaborn`

## 📁 Repository
```
├── AT&T_Ds3.ipynb      # full analysis: EDA, 3 models, comparison
├── spam.csv            # dataset
├── requirements.txt
└── SLIDES              # link to the presentation
```

## ▶️ Run it
```bash
pip install -r requirements.txt
jupyter notebook "AT&T_Ds3.ipynb"
```

---
👤 **Nouman Malidi Mlimi** — Data Scientist / Data Analyst · [Portfolio](https://noumanmalidi7.github.io) · [LinkedIn](https://www.linkedin.com/in/nouman-malidi-a99276326/)
