# 🏥 HCT DATATHON 2025 - CLINICAL INTELLIGENCE PLATFORM
# Author: Your Name
# Institution: Higher Colleges of Technology

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ------------------------- PAGE CONFIG -------------------------
st.set_page_config(page_title="🏥 HCT Datathon 2025", page_icon="💉", layout="wide")

st.markdown("<h1 style='text-align:center; color:#1f77b4;'>🏥 HCT Datathon 2025 - Clinical Intelligence Platform</h1>", unsafe_allow_html=True)
st.markdown("---")

# ------------------------- FILE UPLOAD -------------------------
st.sidebar.header("📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

    # ------------------------- DATA OVERVIEW -------------------------
    st.subheader("📊 Data Overview")
    st.write(df.head())
    st.write("**Data Info:**")
    st.write(df.describe())

    # Handle missing values
    if df.isnull().sum().sum() > 0:
        st.warning("⚠️ Missing values detected! Filling with median/mode.")
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)

    # ------------------------- TARGET SELECTION -------------------------
    target = st.sidebar.selectbox("🎯 Select Target Variable:", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        # Encode categorical features
        X = pd.get_dummies(X, drop_first=True)

        # Encode target if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Split safely
        if y.nunique() > 1:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42
                )
        else:
            st.warning("⚠️ Target has only one class — cannot train models.")
            st.stop()

        # Scale numeric features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ------------------------- MODEL TRAINING -------------------------
        st.markdown("## 🤖 Predictive Modeling")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred, zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
            }
            results.append(metrics)

        results_df = pd.DataFrame(results).round(3)
        st.dataframe(results_df, use_container_width=True)

        # ------------------------- VISUAL INSIGHTS -------------------------
        st.markdown("## 📈 Model Performance Insights")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(results_df, x="Model", y="Accuracy", title="Model Accuracy", color="Model")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(results_df, x="Model", y="F1-Score", title="F1-Score Comparison", color="Model")
            st.plotly_chart(fig, use_container_width=True)

        # Confusion matrix
        st.markdown("### 🧩 Confusion Matrix")
        for name, model in models.items():
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix - {name}")
            st.pyplot(fig)

        # ------------------------- RECOMMENDATIONS -------------------------
        st.markdown("## 💡 AI-Driven Insights & Recommendations")

        best_model = results_df.sort_values(by="F1-Score", ascending=False).iloc[0]
        st.success(f"🏆 **Best Model:** {best_model['Model']} (F1 = {best_model['F1-Score']})")

        st.markdown("""
        **Recommendations:**
        - 🔍 Validate the model using cross-validation for consistency.
        - 🧠 Incorporate more clinical and behavioral data to improve accuracy.
        - ⚖️ Ensure explainability and fairness in AI-based healthcare models.
        - 💉 Deploy the model responsibly with medical expert supervision.
        """)
else:
    st.info("👈 Upload your dataset in the sidebar to begin.")
