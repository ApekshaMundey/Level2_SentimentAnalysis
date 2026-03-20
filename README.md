# 🎬 Sentiment Analysis on Movie Reviews

##  Overview

This project presents a **comprehensive study of sentiment analysis models**, progressing from traditional Machine Learning approaches to Deep Learning and Transformer-based architectures.

The goal is not only to achieve high accuracy but also to **understand model behavior**, evaluate performance across representations, and provide **interpretability using LIME**.

---

## 🚀 Key Highlights

*  Comparative analysis of **ML, DL, and Transformer models**
*  Evaluation across multiple text representations (TF-IDF, embeddings, contextual embeddings)
*  Detailed **error analysis on linguistic challenges**
*  Explainability using **LIME (Local Interpretable Model-Agnostic Explanations)**
*  Interactive **Streamlit web app with highlighted word contributions**
*  Detection of **mixed sentiment / sarcasm patterns**

---

## 🧪 Models & Methodology

The project is structured in **three progressive stages**:

### 🔹 Stage 1: Traditional Machine Learning

* Logistic Regression (TF-IDF)
* Linear SVM (TF-IDF)

### 🔹 Stage 2: Deep Learning

* Dense Neural Network (TF-IDF)
* LSTM with Learned Embeddings
* LSTM with Pre-trained GloVe Embeddings

### 🔹 Stage 3: Transformer-Based Model

* DistilBERT (Contextual Embeddings)

---

## 📊 Performance Comparison

| Level       | Model               | Representation       | Test Accuracy | Overfitting |
| ----------- | ------------------- | -------------------- | ------------- | ----------- |
| ML          | Logistic Regression | TF-IDF               | 92%           | Low         |
| ML          | Linear SVM          | TF-IDF               | 92%           | Low         |
| DL          | Dense NN            | TF-IDF               | 89%           | High        |
| DL          | LSTM                | Learned Embedding    | 86%           | High        |
| DL          | LSTM + GloVe        | Pretrained Embedding | 83%           | Moderate    |
| Transformer | DistilBERT          | Contextual Embedding | 91%           | Low         |

---

## 🔍 Key Observations

* Traditional ML models performed **surprisingly strong** with TF-IDF features.
* Deep Learning models showed **higher overfitting**, especially with limited data.
* Pretrained embeddings (GloVe) did not outperform TF-IDF in this setup.
* Transformer model (DistilBERT) achieved **balanced performance with low overfitting**.
* Contextual embeddings improved generalization compared to static representations.

---

## 🧠 Error Analysis

The models were evaluated on challenging linguistic patterns:

* Mixed sentiment reviews
* Sarcasm and implicit tone
* Long analytical reviews
* Exaggeration and emotional intensity
* Personal narrative reviews

### Insight:

While Transformer models handle context better, **sarcasm remains challenging** due to implicit contradictions and tone.

---

## 💡 Explainability with LIME

To improve transparency, the project integrates:

* **LIME for local explanations**
* Word-level importance visualization
* Color-coded highlighting:

  * 🟩 Positive contribution
  * 🟥 Negative contribution

This helps users understand **why a prediction was made**, not just what the prediction is.

---

## ⚠️ Handling Sarcasm & Mixed Sentiment

A lightweight heuristic was introduced to detect **sentiment shifts**:

* Identifies transitions from positive → negative words
* Flags potential sarcasm or mixed sentiment
* Reduces confidence to reflect uncertainty

---

## 🖥️ Streamlit Application

An interactive web app was built using Streamlit featuring:

* Real-time sentiment prediction
* Confidence scores
* LIME-based explanations
* Highlighted text visualization
* Sarcasm / mixed sentiment warnings

Streamlit deployment link: https://level2sentimentanalysis-lvtgynsgvsmjrmjs63e9dg.streamlit.app/

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* PyTorch
* Transformers (Hugging Face)
* LIME
* Streamlit

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```
Streamlit deployment link: https://level2sentimentanalysis-lvtgynsgvsmjrmjs63e9dg.streamlit.app/

---

## 📈 Future Improvements

* Fine-tuning on sarcasm-specific datasets
* Integration of SHAP for global explanations
* Multi-class sentiment (neutral, mixed)
* Deployment with scalable backend

---

## 🎯 Conclusion

This project demonstrates that:

* Simpler models can remain competitive with strong feature engineering
* Deep Learning is not always superior without sufficient data
* Transformer models provide a strong balance of performance and generalization
* Explainability is essential for building **trustworthy AI systems**

---

## 👤 Author
**Apeksha Mundey**

Developed as part of an applied machine learning and NLP study focusing on **model comparison, interpretability, and real-world behavior analysis**.

---
