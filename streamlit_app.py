import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

st.set_page_config(page_title="Cancer Dataset Model Evaluator", layout="wide")

st.title("Cancer Dataset Multi-Model Evaluator")
st.write("Upload your test dataset (CSV) to select trained models and view evaluation performance.")

# --- Load Scaler and Models ---
@st.cache_resource
def load_artifacts():
    models = {
        "Logistic Regression": joblib.load("models/clf.joblib"),
        "Decision Tree Classifier": joblib.load("models/dtc.joblib"),
        "KNN Classifier": joblib.load("models/knnc.joblib"),
        "Naive Bayes Classifier": joblib.load("models/nbc.joblib"),
        "Random Forest": joblib.load("models/rf.joblib")
    }
    return models

try:
    models = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please place `scaler.joblib` and model `.joblib` files in the working directory.")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Controls & Settings")

# Feature b: Model Selection Dropdown
selected_model_name = st.sidebar.selectbox("Select Model", list(models.keys()))
model = models[selected_model_name]

# Feature a: Dataset Upload Option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df_test.head())

    # Target column selector
    target_col = st.sidebar.selectbox(
        "Select Target Column (Ground Truth)", 
        options=df_test.columns, 
        index=len(df_test.columns) - 1
    )

    if st.sidebar.button("Run Evaluation", type="primary"):
        # Separate features (X) and label (y)
        X_test = df_test.drop(columns=[target_col])
        y_test = df_test[target_col]

        # Preprocess features
        # X_test_scaled = scaler.transform(X_test)

        # Predict
        y_pred = model.predict(X_test)
        
        # Probabilities (if supported)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        st.markdown("---")

        # Feature c: Display Evaluation Metrics
        st.subheader(f"Evaluation Metrics — {selected_model_name}")
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")
        c4.metric("F1-Score", f"{f1:.4f}")

        if y_proba is not None:
            try:
                auc = roc_auc_score(y_test, y_proba)
                c5.metric("ROC-AUC", f"{auc:.4f}")
            except Exception:
                c5.metric("ROC-AUC", "N/A")

        st.markdown("---")

        # Feature d: Confusion Matrix & Classification Report
        col_cm, col_cr = st.columns([1, 1])

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            st.pyplot(fig)

        with col_cr:
            st.subheader("Classification Report")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df.style.format(precision=3))

else:
    st.info("Upload a CSV file using the sidebar to run model evaluations.")
