import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import shap
import plotly.express as px
import plotly.graph_objects as go
import io
import joblib

# -----------------------------
# 1️⃣ Load Data
# -----------------------------
st.title("🏥 Healthcare Analytics Platform")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write(df.head())

    target_col = st.selectbox("Select the target column", df.columns)
    
    # -----------------------------
    # 2️⃣ Preprocess Data
    # -----------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Determine safe number of CV folds
    min_class_count = y_train.value_counts().min()
    cv_folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # -----------------------------
    # 3️⃣ Model Selection
    # -----------------------------
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
        }
    }
    
    selected_models = st.multiselect("Select models to train", list(models_config.keys()), default=list(models_config.keys()))
    use_grid_search = st.checkbox("Use Grid Search for hyperparameter tuning", value=True)
    
    analytics = type("Analytics", (), {})()  # simple object to store results
    analytics.models = {}
    results = []
    
    # -----------------------------
    # 4️⃣ Train Models
    # -----------------------------
    for model_name in selected_models:
        st.write(f"Training {model_name}...")
        model_config = models_config[model_name]
        model = model_config["model"]
        
        if use_grid_search and model_config["params"]:
            gs = GridSearchCV(model, model_config["params"], cv=cv, scoring="f1_macro")
            gs.fit(X_train_scaled, y_train)
            best_model = gs.best_estimator_
            st.success(f"Best params for {model_name}: {gs.best_params_}")
        else:
            best_model = model
            best_model.fit(X_train_scaled, y_train)
        
        analytics.models[model_name] = best_model
        
        # Predictions
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Metrics
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")
        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            "CV F1-Mean": cv_scores.mean(),
            "CV F1-Std": cv_scores.std()
        }
        results.append(metrics)
    
    analytics.results = pd.DataFrame(results)
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(analytics.results)
    
    # -----------------------------
    # 5️⃣ SHAP Explainability
    # -----------------------------
    st.subheader("🧮 SHAP Explainability")
    explain_model_name = st.selectbox("Select model for SHAP", list(analytics.models.keys()))
    model = analytics.models[explain_model_name]
    
    # Take small sample for SHAP
    X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
    X_sample_encoded = pd.get_dummies(X_sample, drop_first=True)
    
    try:
        explainer = shap.Explainer(model, X_sample_encoded)
        shap_values = explainer(X_sample_encoded)
        st.subheader("SHAP Summary Plot")
        shap.summary_plot(shap_values.values, X_sample_encoded, show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.warning(f"SHAP could not run: {e}")
    
    # -----------------------------
    # 6️⃣ Prescriptive Insights
    # -----------------------------
    st.subheader("💡 Prescriptive Insights")
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame({
            "feature": X_sample_encoded.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        st.bar_chart(feature_importance.head(10).set_index("feature"))
    
    # -----------------------------
    # 7️⃣ Download Results
    # -----------------------------
    st.subheader("📤 Download Results")
    csv_buffer = io.BytesIO()
    analytics.results.to_csv(csv_buffer, index=False)
    st.download_button("Download Metrics CSV", data=csv_buffer.getvalue(), file_name="metrics.csv", mime="text/csv")
    
    # Best model download
    best_model_name = analytics.results.loc[analytics.results['F1-Score'].idxmax(), 'Model']
    best_model = analytics.models[best_model_name]
    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    st.download_button("Download Best Model", data=model_buffer.getvalue(), file_name=f"best_model_{best_model_name}.joblib", mime="application/octet-stream")
