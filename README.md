# Health Condition Predictor using Few-Shot Learning and TF-IDF Baseline

## 📚 Overview

This project explores the application of **Few-Shot Learning** using **SetFit** models in the **healthcare domain**, specifically for the task of **text classification** based on symptom descriptions. The system predicts possible health conditions from user-reported symptoms extracted from Reddit posts.

We compare a recent NLP technique (SetFit) with a traditional TF-IDF + Logistic Regression baseline to evaluate the effectiveness of few-shot learning on small, domain-specific datasets.

---

## 🎯 Objectives

- Apply **Few-Shot Learning** to the healthcare domain using real-world social media data.
- Build a web-based application using **Streamlit** that predicts health conditions from free-text symptoms.
- Compare model performance between:
  - A traditional **TF-IDF + Logistic Regression** baseline.
  - A **SetFit-based few-shot learning** approach.
- Offer basic model explainability for user input predictions.

---

## 🧠 NLP Techniques

- **Few-Shot Learning** with [SetFit](https://github.com/huggingface/setfit): Efficient fine-tuning of sentence transformers without large-scale data.
- **Explainable AI**: Basic interpretability using TF-IDF top features.
- **Baseline Model**: TF-IDF feature extraction with a Logistic Regression classifier.

---

## 🩺 Domain: Healthcare

### Task: Symptom-Based Text Classification

Users input symptom descriptions. The model classifies these into predefined health condition categories based on Reddit health-related posts.

---

## 🛠️ Technologies Used

- **Python**, **Streamlit**
- **scikit-learn** for baseline model
- **Hugging Face SetFit** for few-shot learning
- **Seaborn & Matplotlib** for data visualization
- **Google Colab + Drive** for training environment

---

## 📁 Dataset

- **Source**: Pre-annotated Reddit health-related posts.
- **Format**: CSV with `text` and `labels` columns.
- **Preprocessing**:
  - Clean text (remove URLs, special characters, lowercase).
  - Filter short entries.
  - Balance data using `samples_per_class` sampling.

---

## 📊 Evaluation

Both models are evaluated on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrix**

> SetFit achieves strong performance even with minimal data, demonstrating its usefulness in low-resource domains like medical text classification.

---

## 🚀 How to Run

### Setup (on Google Colab)

1. Upload `reddit_posts_raw_data.csv` to your **Google Drive**.
2. Mount your drive in Google Colab:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
