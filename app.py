# -----------------------------
# 1️⃣ Prepare CV folds safely
# -----------------------------
from sklearn.model_selection import StratifiedKFold

# Determine minimum class count
min_class_count = y_train.value_counts().min()
cv_folds = min(5, min_class_count)  # max 5 folds or fewer if class small
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

if min_class_count < 2:
    st.warning("⚠️ Some classes have fewer than 2 samples. Model training may fail or overfit.")

# -----------------------------
# 2️⃣ Define Models & Parameters
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models_config = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10]
        }
    },
    "SVM": {
        "model": SVC(probability=True),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }
}

# -----------------------------
# 3️⃣ Train Models
# -----------------------------
results = []
analytics.models = {}

for model_name, config in models_config.items():
    st.write(f"Training {model_name}...")

    model = config["model"]
    params = config.get("params", None)

    if use_grid_search and params:
        try:
            gs = GridSearchCV(model, params, cv=cv, scoring="f1_macro")
            gs.fit(X_train_scaled, y_train)
            best_model = gs.best_estimator_
            st.success(f"Best params for {model_name}: {gs.best_params_}")
        except Exception as e:
            st.error(f"GridSearchCV failed for {model_name}: {e}")
            best_model = model
            best_model.fit(X_train_scaled, y_train)
    else:
        best_model = model
        best_model.fit(X_train_scaled, y_train)

    # Store model
    analytics.models[model_name] = best_model

    # Predict & evaluate
    y_pred = best_model.predict(X_test_scaled)
    y_prob = best_model.predict_proba(X_test_scaled)[:, 1] if hasattr(best_model, "predict_proba") else None

    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv, scoring="f1_macro")

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1-Score": f1_score(y_test, y_pred, average="weighted"),
        "ROC-AUC": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
        "CV F1-Mean": cv_scores.mean(),
        "CV F1-Std": cv_scores.std()
    })

analytics.results = pd.DataFrame(results)
st.success("✅ All models trained successfully!")

# -----------------------------
# 4️⃣ Prescriptive Insights
# -----------------------------
best_model_name = analytics.results.loc[analytics.results['F1-Score'].idxmax(), 'Model']
best_model = analytics.models[best_model_name]
st.success(f"Using {best_model_name} for prescriptive insights")

# Feature importance
if hasattr(best_model, "feature_importances_"):
    X = df.drop(columns=[analytics.target_col])
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values("importance", ascending=False)
    st.write("Top Features:", feature_importance.head(5))
