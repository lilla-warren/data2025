# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import joblib
from datetime import datetime
import io

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)

# Initialize analytics storage
class Analytics:
    def __init__(self):
        self.models = {}
        self.results = pd.DataFrame()
        self.target_col = None

analytics = Analytics()

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="Healthcare Analytics Platform", layout="wide")
st.title("🏥 HCT Datathon 2025 - Healthcare Analytics Platform")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.write("Dataset Preview:")
    st.sidebar.dataframe(df.head())
    
    # Target selection
    analytics.target_col = st.sidebar.selectbox("Select target variable", df.columns)
    
    X = df.drop(columns=[analytics.target_col])
    y = df[analytics.target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Scaling numeric features
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Sidebar: Model selection & options
    st.sidebar.header("⚙️ Modeling Options")
    selected_models = st.sidebar.multiselect(
        "Select models to train",
        ["Logistic Regression", "Random Forest", "SVM"],
        default=["Logistic Regression", "Random Forest"]
    )
    use_grid_search = st.sidebar.checkbox("Enable Grid Search for hyperparameter tuning", value=True)
    
    # ----------------------
    # 1️⃣ Train models
    # ----------------------
    if st.sidebar.button("Train Models"):
        st.subheader("🚀 Training Models...")
        
        models_config = {
            "Logistic Regression": {
                "model": LogisticRegression(max_iter=1000),
                "params": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
            },
            "Random Forest": {
                "model": RandomForestClassifier(random_state=42),
                "params": {"n_estimators": [50, 100, 200], "max_depth": [None, 5, 10]}
            },
            "SVM": {
                "model": SVC(probability=True),
                "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
            }
        }
        
        results = []
        
        for model_name in selected_models:
            st.write(f"Training {model_name}...")
            model_config = models_config[model_name]
            model = model_config["model"]
            
            # Grid Search
            if use_grid_search and model_config["params"]:
                gs = GridSearchCV(model, model_config["params"], cv=5, scoring="f1_macro")
                gs.fit(X_train_scaled, y_train)
                best_model = gs.best_estimator_
                st.success(f"{model_name} best params: {gs.best_params_}")
            else:
                best_model = model
                best_model.fit(X_train_scaled, y_train)
            
            analytics.models[model_name] = best_model
            
            # Predictions & metrics
            y_pred = best_model.predict(X_test_scaled)
            y_prob = best_model.predict_proba(X_test_scaled)[:,1] if hasattr(best_model, "predict_proba") else None
            cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring="f1_macro")
            
            metrics = {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average='weighted'),
                "Recall": recall_score(y_test, y_pred, average='weighted'),
                "F1-Score": f1_score(y_test, y_pred, average='weighted'),
                "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
                "CV F1-Mean": cv_scores.mean(),
                "CV F1-Std": cv_scores.std()
            }
            results.append(metrics)
        
        analytics.results = pd.DataFrame(results)
        st.success("✅ Models trained successfully!")
        
        # Display results
        st.subheader("📊 Model Performance Comparison")
        st.dataframe(analytics.results.style.format({
            'Accuracy':'{:.3f}','Precision':'{:.3f}','Recall':'{:.3f}',
            'F1-Score':'{:.3f}','ROC-AUC':'{:.3f}','CV F1-Mean':'{:.3f}','CV F1-Std':'{:.3f}'
        }).highlight_max(subset=['Accuracy','F1-Score','ROC-AUC'], color='lightgreen'), use_container_width=True)
        
        # ----------------------
        # 2️⃣ ROC Curves
        # ----------------------
        st.subheader("📈 ROC Curves")
        fig = go.Figure()
        for model_name in selected_models:
            if model_name in analytics.models:
                model = analytics.models[model_name]
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)[:,1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc_score = roc_auc_score(y_test, y_prob)
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={auc_score:.3f})"))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash='dash', color='gray'), name='Random'))
        fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig, use_container_width=True)
        
        # ----------------------
        # 3️⃣ Confusion Matrix
        # ----------------------
        st.subheader("🧮 Confusion Matrix")
        selected_cm_model = st.selectbox("Select model", selected_models)
        if selected_cm_model in analytics.models:
            model = analytics.models[selected_cm_model]
            y_pred = model.predict(X_test_scaled)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual", color="Count"),
                               x=['Class 0','Class 1'], y=['Class 0','Class 1'], title=f"Confusion Matrix - {selected_cm_model}")
            st.plotly_chart(fig_cm, use_container_width=True)
        
        # ----------------------
        # 4️⃣ SHAP Explainability
        # ----------------------
        st.subheader("🔍 SHAP Explainability")
        explain_model_name = st.selectbox("Select model for SHAP", selected_models)
        if explain_model_name in analytics.models:
            model = analytics.models[explain_model_name]
            X_encoded = pd.get_dummies(X, drop_first=True)
            X_sample = X_encoded.sample(min(100, len(X_encoded)), random_state=42)
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample)
            # Summary plot
            fig_shap, ax = plt.subplots(figsize=(10,6))
            shap.summary_plot(shap_values.values, X_sample, show=False)
            st.pyplot(fig_shap)
        
        # ----------------------
        # 5️⃣ Prescriptive Insights
        # ----------------------
        st.subheader("💡 Prescriptive Insights")
        best_model_name = analytics.results.loc[analytics.results['F1-Score'].idxmax(), 'Model']
        best_model = analytics.models[best_model_name]
        st.success(f"Using {best_model_name} for insights")
        
        shap_importance = pd.DataFrame({
            "feature": X_encoded.columns,
            "importance": np.abs(shap_values.values).mean(0)
        }).sort_values("importance", ascending=False)
        
        st.write("Top 5 features for intervention:")
        st.dataframe(shap_importance.head(5))
        
        # ----------------------
        # 6️⃣ Download Results & Model
        # ----------------------
        st.sidebar.header("📤 Export Results")
        csv_buffer = io.BytesIO()
        analytics.results.to_csv(csv_buffer, index=False)
        st.sidebar.download_button("Download Metrics CSV", csv_buffer.getvalue(), "model_metrics.csv", "text/csv")
        
        model_buffer = io.BytesIO()
        joblib.dump(best_model, model_buffer)
        st.sidebar.download_button("Download Best Model", model_buffer.getvalue(), f"best_model_{best_model_name}.joblib", "application/octet-stream")

else:
    st.markdown("""
    ## 🏥 Welcome to HCT Datathon 2025 Healthcare Analytics Platform
    Upload your dataset to start analysis!
    """)

if __name__ == "__main__":
    pass
