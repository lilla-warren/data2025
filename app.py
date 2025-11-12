import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib
import io

# -------------------------------------
# 🎯 PAGE SETUP
# -------------------------------------
st.set_page_config(page_title="🏥 Healthcare Analytics Platform", layout="wide")

st.title("🏥 Advanced Healthcare Analytics Platform")
st.write("A unified platform for descriptive, diagnostic, predictive, and prescriptive healthcare analytics — fully deployable via GitHub + Streamlit.")

# -------------------------------------
# 📂 1. DATA UPLOAD
# -------------------------------------
uploaded_file = st.file_uploader("📤 Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")
    st.write("**Data Preview:**")
    st.dataframe(df.head())

    target_col = st.selectbox("🎯 Select the Target Column", df.columns)

    # -------------------------------------
    # 🧹 2. DATA PREPROCESSING
    # -------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    min_class_count = y_train.value_counts().min()
    cv_folds = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # -------------------------------------
    # 🤖 3. MODELING
    # -------------------------------------
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1, 10]}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
        }
    }

    selected_models = st.multiselect("Select Models to Train", list(models_config.keys()), default=list(models_config.keys()))
    use_grid = st.checkbox("Use Grid Search for Hyperparameter Tuning", True)

    results = []
    trained_models = {}

    st.subheader("⚙️ Model Training & Evaluation")
    progress = st.progress(0)
    step = 1 / len(selected_models)

    for i, model_name in enumerate(selected_models):
        model_cfg = models_config[model_name]
        model = model_cfg["model"]

        if use_grid:
            gs = GridSearchCV(model, model_cfg["params"], cv=cv, scoring="f1_macro")
            gs.fit(X_train_scaled, y_train)
            best_model = gs.best_estimator_
            st.write(f"✅ {model_name}: Best Params →", gs.best_params_)
        else:
            best_model = model.fit(X_train_scaled, y_train)

        trained_models[model_name] = best_model

        # Predictions & Metrics
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

        cv_score = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")

        metrics = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted"),
            "Recall": recall_score(y_test, y_pred, average="weighted"),
            "F1-Score": f1_score(y_test, y_pred, average="weighted"),
            "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
            "CV F1 Mean": cv_score.mean()
        }
        results.append(metrics)
        progress.progress((i + 1) * step)

    results_df = pd.DataFrame(results)
    st.dataframe(results_df, use_container_width=True)

    # -------------------------------------
    # 📈 4. ROC CURVES
    # -------------------------------------
    st.subheader("📈 ROC Curves")
    fig = go.Figure()
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------
    # 🧮 5. CONFUSION MATRIX
    # -------------------------------------
    st.subheader("🧮 Confusion Matrix")
    selected_cm_model = st.selectbox("Select model for Confusion Matrix", list(trained_models.keys()))
    cm_model = trained_models[selected_cm_model]
    cm = confusion_matrix(y_test, cm_model.predict(X_test_scaled))
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", labels=dict(x="Predicted", y="Actual"))
    st.plotly_chart(fig_cm, use_container_width=True)

    # -------------------------------------
    # 🔍 6. SHAP EXPLAINABILITY
    # -------------------------------------
    st.subheader("🔍 SHAP Explainability")
    shap_model = st.selectbox("Select model for SHAP analysis", list(trained_models.keys()))
    model = trained_models[shap_model]

    try:
        X_sample = pd.DataFrame(X_train_scaled, columns=X.columns).sample(80, random_state=42)
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explainability unavailable: {e}")

    # -------------------------------------
    # 💡 7. PRESCRIPTIVE INSIGHTS
    # -------------------------------------
    st.subheader("💡 Prescriptive Insights & Recommendations")

    best_model_name = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    best_model = trained_models[best_model_name]

    if hasattr(best_model, "feature_importances_"):
        feat_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)
        st.write(f"**Top Contributing Features ({best_model_name})**")
        st.bar_chart(feat_importance.set_index("Feature").head(10))

        st.markdown("### 📋 Recommendations:")
        for feat in feat_importance.head(5)["Feature"]:
            if "blood" in feat.lower():
                st.markdown(f"- `{feat}`: Encourage regular blood pressure monitoring.")
            elif "glucose" in feat.lower():
                st.markdown(f"- `{feat}`: Recommend lifestyle changes for blood sugar control.")
            elif "age" in feat.lower():
                st.markdown(f"- `{feat}`: Target preventive health programs for older populations.")
            elif "bmi" in feat.lower():
                st.markdown(f"- `{feat}`: Promote physical activity and nutrition awareness.")
            else:
                st.markdown(f"- `{feat}`: Continuous monitoring recommended for early risk detection.")

    # -------------------------------------
    # 📦 8. DOWNLOAD RESULTS
    # -------------------------------------
    st.subheader("📦 Download Results & Model")

    buffer = io.BytesIO()
    results_df.to_csv(buffer, index=False)
    st.download_button("⬇️ Download Metrics CSV", data=buffer.getvalue(), file_name="model_metrics.csv")

    model_buffer = io.BytesIO()
    joblib.dump(best_model, model_buffer)
    st.download_button("⬇️ Download Best Model", data=model_buffer.getvalue(), file_name=f"best_{best_model_name}.joblib")
