# ================== IMPORTS ==================
import streamlit as st
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import seaborn as sns
import matplotlib.pyplot as plt
import os


# TEXT PREPROCESSING 
def clean_text(text):
  
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip()  # Convert to lowercase and remove leading/trailing spaces
    return text


# DATA LOADING
@st.cache_data
def load_data(file_path, samples_per_class=70):
    
    try:
        df = pd.read_csv(file_path)
        df['text'] = df['text'].apply(clean_text)
        df = df[df['text'].str.len() > 10]  # Filter out short/noisy samples

        # Sample balanced classes
        sampled_df = df.groupby('labels').apply(
            lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=42)
        ).reset_index(drop=True)

        st.success(f"Dataset loaded and sampled: {len(sampled_df)} samples across {len(sampled_df['labels'].unique())} labels.")
        return sampled_df
    except FileNotFoundError:
        st.error(f"Dataset file not found at: {file_path}. Please ensure it's in the correct path.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# BASELINE MODEL (TF-IDF + LOGISTIC REGRESSION)
def train_baseline_model(X_train, y_train, X_test, y_test):
    """
    Train a baseline model using TF-IDF vectorizer and Logistic Regression.
    """
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500, C=1.0, solver='liblinear')
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[0],
        'recall': precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[1],
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return model, vectorizer, metrics, y_pred


# FEW-SHOT LEARNING MODEL (SETFIT)
def train_setfit_model(X_train, y_train, X_test, y_test):
    """
    Train a Few-Shot Learning model using SetFit with contrastive learning.
    """
    train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    test_dataset = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})

    model = SetFitModel.from_pretrained("sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

    trainer = SetFitTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        num_iterations=20,
        num_epochs=5,
        batch_size=8,
        learning_rate=2e-5,
        seed=42,
        report_to="none"
    )
    trainer.train()

    y_pred = model.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[0],
        'recall': precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[1],
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return model, metrics, y_pred


# CONFUSION MATRIX VISUALIZATION
def plot_confusion_matrix(y_true, y_pred, labels, title):
    """
    Plot a confusion matrix using Seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)


# MAIN STREAMLIT APP 
def main():
    st.title("🩺 Health Condition Predictor")
    st.write("This app uses machine learning models trained on Reddit health data to predict possible medical conditions from symptoms.")
    st.markdown("---")

    # Path to dataset 
    DATA_FILE_PATH = '/content/drive/MyDrive/NLP_Project/reddit_posts_raw_data.csv'

    # Load and sample dataset
    df = load_data(DATA_FILE_PATH, samples_per_class=70)
    if df is None:
        st.warning("Please address the data loading issue to proceed.")
        return

    X = df['text'].values
    y = df['labels'].values
    unique_labels = np.unique(y)

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    st.info(f"Training data size: {len(X_train)} | Test data size: {len(X_test)}")

    # Display training progress
    st.subheader("Model Training Progress")
    progress_bar = st.progress(0)

    # Train baseline model
    with st.spinner("Training baseline model (TF-IDF + Logistic Regression)..."):
        baseline_model, vectorizer, baseline_metrics, baseline_y_pred = train_baseline_model(X_train, y_train, X_test, y_test)
        progress_bar.progress(50)

    # Train SetFit Few-Shot Learning model
    with st.spinner("Training Few-Shot Learning model (SetFit) - This might take longer..."):
        setfit_model, setfit_metrics, setfit_y_pred = train_setfit_model(X_train, y_train, X_test, y_test)
        progress_bar.progress(100)

    st.success("✅ Models trained successfully!")

    # Display model performance
    st.subheader("📊 Model Performance Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Baseline (TF-IDF + Logistic Regression)**")
        st.write(f"Accuracy: {baseline_metrics['accuracy']:.3f}")
        st.write(f"F1-Score: {baseline_metrics['f1']:.3f}")
        st.write(f"Precision: {baseline_metrics['precision']:.3f}")
        st.write(f"Recall: {baseline_metrics['recall']:.3f}")
    with col2:
        st.write("**Few-Shot Learning (SetFit)**")
        st.write(f"Accuracy: {setfit_metrics['accuracy']:.3f}")
        st.write(f"F1-Score: {setfit_metrics['f1']:.3f}")
        st.write(f"Precision: {setfit_metrics['precision']:.3f}")
        st.write(f"Recall: {setfit_metrics['recall']:.3f}")

    st.markdown("---")

    # Confusion matrix visualizations
    st.subheader("🧩 Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        plot_confusion_matrix(y_test, baseline_y_pred, unique_labels, "Baseline Confusion Matrix")
    with col2:
        plot_confusion_matrix(y_test, setfit_y_pred, unique_labels, "SetFit Confusion Matrix")

    st.markdown("---")

    # Real-time prediction section
    st.subheader("🤖 Predict Health Condition from Symptoms")
    user_input = st.text_area("Describe your symptoms:", height=150)

    if st.button("Predict"):
        if user_input:
            cleaned_input = clean_text(user_input)
            if len(cleaned_input) < 10:
                st.warning("Please enter a more detailed symptom description (at least 10 characters).")
                return

            # Baseline prediction
            input_tfidf = vectorizer.transform([cleaned_input])
            baseline_pred = baseline_model.predict(input_tfidf)[0]
            baseline_proba = baseline_model.predict_proba(input_tfidf)[0]
            baseline_confidence = max(baseline_proba) * 100

            # SetFit prediction
            setfit_pred = setfit_model.predict([cleaned_input])[0]
            setfit_proba = setfit_model.predict_proba([cleaned_input])[0]
            setfit_confidence = max(setfit_proba) * 100

            # Display predictions
            st.write("### 🔍 Predictions")
            st.write(f"**Baseline Model:** `{baseline_pred}` (Confidence: {baseline_confidence:.2f}%)")
            st.write(f"**Few-Shot Model (SetFit):** `{setfit_pred}` (Confidence: {setfit_confidence:.2f}%)")

            # Baseline explainability using TF-IDF
            st.write("### 📌 Explanation (Baseline Model)")
            try:
                feature_names = vectorizer.get_feature_names_out()
                coef_for_predicted_class = baseline_model.coef_[baseline_model.classes_.tolist().index(baseline_pred)]
                top_feature_indices = np.argsort(coef_for_predicted_class)[-5:]

                st.write("Top contributing words:")
                for idx in top_feature_indices:
                    st.write(f"- `{feature_names[idx]}` (Weight: {coef_for_predicted_class[idx]:.3f})")
            except Exception as e:
                st.info(f"Could not provide detailed explanation for baseline model: {e}")
        else:
            st.warning("Please enter a symptom description.")

  


# RUN APP
if __name__ == "__main__":
    main()
