Health Condition Predictor: Few-Shot Learning for Symptom Classification
Project Overview
This project is part of the Natural Language Processing (AIE241) extracurricular activity at Alamein International University. It explores Few-Shot Learning using the SetFit model to predict health conditions from user-provided symptom descriptions, applied to the healthcare domain. The system uses a dataset of Reddit posts to classify symptoms into health conditions, comparing a Few-Shot Learning model (SetFit with distilbert-base-uncased) against a baseline TF-IDF + Logistic Regression model. A Streamlit interface allows users to input symptoms and receive predictions with confidence scores and basic explainability (top TF-IDF features for the baseline model).
Hypothesis
Does Few-Shot Learning (using SetFit) outperform a TF-IDF + Logistic Regression baseline for health condition prediction from Reddit symptom descriptions in terms of accuracy and F1-score?
Dataset

Source: The dataset (reddit_posts_raw_data.csv) is a custom collection of Reddit posts related to health symptoms and their associated conditions. It is assumed to be sourced from public Reddit data (e.g., via APIs or Kaggle, not included in this repository due to size constraints).
Structure: The dataset contains two main columns:
text: User-described symptoms (free-form text).
labels: Health condition labels (e.g., "flu", "migraine").


Preprocessing: 
Text cleaning: Remove URLs, special characters, and convert to lowercase.
Filtering: Exclude texts shorter than 10 characters.
Sampling: Balanced to 30 samples per class to reduce computational load.


Note: To replicate this project, obtain a similar dataset from sources like Kaggle or Reddit APIs, ensuring it has text and labels columns.

Requirements

Python: 3.8 or higher
Dependencies:pip install streamlit pandas numpy scikit-learn setfit datasets seaborn matplotlib


Dataset: Place reddit_posts_raw_data.csv in the project root directory.

Installation

Clone this repository:git clone <repository-url>
cd <repository-directory>


Install dependencies:pip install -r requirements.txt

Alternatively, install individually:pip install streamlit pandas numpy scikit-learn setfit datasets seaborn matplotlib


Ensure reddit_posts_raw_data.csv is in the project root directory.

Usage

Run the Streamlit app:streamlit run health_condition_predictor.py


Open the provided URL (e.g., http://localhost:8501) in a web browser.
Features:
View model performance metrics (accuracy, F1-score, precision, recall) for both models.
Visualize confusion matrices for baseline and SetFit models.
Enter symptom descriptions in the text area to predict health conditions, with confidence scores and top contributing words (for the baseline model).



Code Structure

health_condition_predictor.py: Main script containing:
Data loading and preprocessing (load_data, clean_text).
Baseline model training (TF-IDF + Logistic Regression).
Few-Shot Learning model training (SetFit with distilbert-base-uncased).
Streamlit interface for predictions and visualizations.


reddit_posts_raw_data.csv: Dataset (not included; must be sourced separately).


Team members collaborated equally on experimentation, metric calculations, and repository setup, tracked via GitHub commits.
Results

Metrics: Both models are evaluated on accuracy, F1-score, precision, and recall (weighted averages).
Visualizations: Confusion matrices are displayed in the Streamlit app.
Explainability: Top TF-IDF features are shown for baseline model predictions.
Sample Output:
Input: "I have a fever and cough."
Baseline Prediction: "Flu" (Confidence: 85.2%)
SetFit Prediction: "Flu" (Confidence: 90.1%)



Limitations

Small dataset size (30 samples per class) may limit model generalization.
Limited explainability for the SetFit model; only the baseline model provides feature weights.
No explicit error analysis (e.g., misclassified examples) is included in the app.

Future Work

Expand the dataset with more samples and diverse health conditions.
Add explainability for the SetFit model using techniques like SHAP or attention visualization.
Conduct error analysis to identify and address misclassifications.
Explore additional Few-Shot Learning techniques or larger transformer models.

Contact
For questions, contact the team via the course Canvas page or GitHub issues.
