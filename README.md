# Sonar: Rock vs Mine Classification

This project implements a machine learning model to classify sonar data as either a **Rock** or a **Mine**.

## 🚀 Overview

The project uses a `RandomForestClassifier` to analyze 60 sonar features and predict the nature of the detected object.

## 📊 Model Performance

The model was evaluated on a test set with the following performance metrics:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 80.95% |
| **Precision** | 79.26% |
| **Recall** | 79.26% |
| **F1 Score** | 79.26% |

## 🛠️ Getting Started

1.  **Environment Setup**: Install the necessary dependencies (ensure you have `scikit-learn`, `pandas`, `joblib`, and `streamlit` installed).
2.  **Train the Model**: Run the training script to generate the `prediction.pkl` file:
    ```bash
    python main.py
    ```
3.  **Run the Application**: Launch the Streamlit web interface:
    ```bash
    streamlit run streamlit_app.py
    ```
## Dataset

[Link](https://www.kaggle.com/datasets/taseermehboob9/submarine-sonar-rock-or-mine-dataset)