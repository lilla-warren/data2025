# app.py - HCT Datathon 2025: Clinical Intelligence Platform
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# --------------------------
# APP CONFIG
# --------------------------
st.set_page_config(page_title="🏥 Clinical Intelligence Platform", layout="wide")

st.title("🏥 HCT Datathon 2025 - Clinical Intelligence Platform")
st.markdown("""
### Transforming Health Data into Knowledge  
**Goal:** Apply machine learning & ethical AI to derive insights, predictions, and recommendations.
---
""")

# --------------------------
# FILE UPLOAD
# --------------------------
st.sidebar.header("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Overview")
    st.dataframe(df.head())
    st.write(f"Shape: {df.shape}")
    st.write("Summary Statistics:")
    st.dataframe(df.describe())

    # --------------------------
    # DESCRIPTIVE ANALYTICS
    # --------------------------
    st.markdown("## 1️⃣ Descriptive Analytics")
    st.write("Visualizing feature distributions and class balance.")
    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df[numeric_cols[0]], kde=True, ax=ax)
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            sns.boxplot(df[numeric_cols[0]], ax=ax)
            st.pyplot(fig)

    # --------------------------
    # CORRELATION (DIAGNOSTIC)
    # --------------------------
    st.markdown("## 2️⃣ Diagnostic Analytics")
    st.write("Exploring feature relationships via correlation heatmap.")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # --------------------------
    # TARGET SELECTION
    # --------------------------
    st.markdown("## 3️⃣ Predictive Modeling")
    target = st.selectbox("Select the target column (label):", df.columns)

    if target:
        X = df.drop(columns=[target])
        y = df[target]

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --------------------------
        # MODEL TRAINING
        # --------------------------
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        results = []
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else y_pred

            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted"),
                "Recall": recall_score(y_test, y_pred, average="weighted"),
                "F1": f1_score(y_test, y_pred, average="weighted"),
                "ROC-AUC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else np.nan
            }
            results.append(metrics)

        results_df = pd.DataFrame(results)
        st.write("### Model Performance Summary")
        st.dataframe(results_df.style.highlight_max(color='lightgreen', axis=0))

        # --------------------------
        # CONFUSION MATRIX & ROC
        # --------------------------
        best_model_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        best_model = models[best_model_name]

        st.markdown(f"### 🎯 Best Model: **{best_model_name}**")
        y_pred_best = best_model.predict(X_test_scaled)

        cm = confusion_matrix(y_test, y_pred_best)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        plt.title("Confusion Matrix")
        st.pyplot(fig)

        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            st.pyplot(fig)

        # --------------------------
        # EXPLAINABILITY (SHAP)
        # --------------------------
        st.markdown("## 4️⃣ Explainability & Transparency")
        try:
            explainer = shap.Explainer(best_model, X_train_scaled)
            shap_values = explainer(X_test_scaled)
            st.write("### SHAP Feature Importance")
            shap.summary_plot(shap_values, X_test, show=False)
            st.pyplot(bbox_inches='tight')
        except Exception as e:
            st.warning(f"SHAP could not run: {e}")

        # --------------------------
        # INSIGHTS & RECOMMENDATIONS
        # --------------------------
        st.markdown("## 5️⃣ Insights & Prescriptive Recommendations")

        top_features = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False) if hasattr(best_model, "feature_importances_") else pd.Series([], dtype=float)
        if not top_features.empty:
            st.write("### 🔍 Top Contributing Features")
            st.bar_chart(top_features.head(5))

            st.write("### 💡 Recommendations")
            st.markdown(f"""
            - Focus on top drivers like **{top_features.index[0]}** for early risk prediction.  
            - Use **{best_model_name}** for production due to its superior accuracy.  
            - Consider data balancing or domain expert input if model bias is detected.  
            - Continuous retraining can enhance fairness and adaptiveness.  
            """)
        else:
            st.info("No feature importances available for this model type.")
else:
    st.info("Please upload a dataset to begin analysis.")
