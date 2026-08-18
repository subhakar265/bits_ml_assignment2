# Cancer Classification Machine Learning Project Documentation

## a. Problem Statement
The purpose of this project is to develop a Machine Learning (ML) classification models for cancer diagnosis using patient data. Early identification of tumors is critical for effective clinical treatment and improving patient health.

---

## b. Dataset Description
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Target Variable:** `diagnosis` (Binary: `1 = Malignant`, `0 = Benign`)
* **Number of Records:** 569 records
* **Number of Features:** 30 features

---

## c. GitHub Repository Link
* **Repository URL:** `https://github.com/subhakar265/bits_ml_assignment2`
* **Repository Structure & Required Files:**
  * `streamlit_app.py`: Streamlit web app supporting CSV upload, model selection, metric calculation, confusion matrix, and classification report.
  * `2025AC05587.ipynb`: Jupyter notebook containing model and evaluation.
  * `models/`: Folder containing saved model artifacts (`.joblib` and `.ipynb`).
  * `data/Cancer_Data.csv`: Sample test dataset for Streamlit app evaluation.
  * `requirements.txt`: List of Python packages required for deployment.
  * `README.md`: Project documentation and model comparison report.

---

## d. Models Used

### 1. Model Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | Precision | Recall | F1 | MCC | ROC_AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | `92.28` | `93.07` | `92.98` | `92.90` | `.85` | `.955` |
| **Decision Tree** | `91.23` | `91.23` | `91.23` | `91.23` | `.81` | `.9058` |
| **kNN** | `96.49` | `96.68` | `96.49` | `96.45` | `.93` | `.9765` |
| **Naive Bayes** | `91.23` | `91.27` | `91.23` | `91.13` | `.81` | `.9335` |
| **Random Forest (Ensemble)** | `97.37` | `97.37` | `97.37` | `97.36` | `.94` | `.9979` |

---

### 2. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | `[Serves as a strong linear baseline. perform well for linear data]` |
| **Decision Tree** | `[ Highly interpretable decision boundaries, but the model overfits the data if the depth is too high.]` |
| **kNN** | `[Strong contender among all the models. Performance depends heavily on value of k. ]` |
| **Naive Bayes** | `[Least performed model with lowest accuracy among all the models. But faster to train]` |
| **Random Forest (Ensemble)** | `[ Reduces variance High stability and low susceptibility to overfitting.]` |

---

### 3. Overall Winner for Your Dataset

* **Overall Winner Model:** `Random Forest`
* **Justification:**
  `Provide justification here. Example: Random Forest achieved the highest performance across key metrics, particularly Recall. In medical cancer diagnostics, minimizing False Negatives (maximizing Recall) is critical so that potential malignant cases are not missed. With an F1-score of [97.36] and MCC of [0.94], this model demonstrated the best balance of precision, sensitivity, and robust generalization on test data.`