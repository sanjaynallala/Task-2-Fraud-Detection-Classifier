
"""
Task 2: Credit Card Fraud Detection
Model: XGBoost Classifier
Dataset: credit_card_fraud_detection_2026.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    average_precision_score
)

# 1. Load dataset
df = pd.read_csv("credit_card_fraud_detection_2026.csv")

print("Dataset shape:", df.shape)

# 2. Prepare target
df = df.dropna(subset=["is_fraud"]).copy()
df["is_fraud"] = df["is_fraud"].astype(int)

print("\nFraud distribution:")
print(df["is_fraud"].value_counts())

# 3. Remove identifiers and target leakage columns
drop_cols = [
    "transaction_id",
    "customer_id",
    "card_id",
    "merchant_id",
    "transaction_datetime",
    "is_fraud",
    "fraud_type",
    "fraud_amount_lost"
]

X = df.drop(columns=drop_cols, errors="ignore")
y = df["is_fraud"]

# 4. Identify numeric and categorical features
numeric_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# 5. Handle missing values and encode categories
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# 6. Split dataset: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 7. Handle imbalanced classes
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_weight = negative_count / positive_count

print("\nTraining legitimate transactions:", negative_count)
print("Training fraudulent transactions:", positive_count)
print("Class imbalance weight:", round(scale_weight, 2))

# 8. Create XGBoost pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_weight,
        random_state=42,
        n_jobs=1,
        tree_method="hist"
    ))
])

# 9. Tune hyperparameters
param_distributions = {
    "classifier__n_estimators": [150, 250, 350],
    "classifier__max_depth": [3, 4, 6],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
    "classifier__min_child_weight": [1, 3, 5]
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="average_precision",
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

print("\nTraining and tuning XGBoost...")
search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation PR-AUC:")
print(round(search.best_score_, 4))

# 10. Predict fraud probabilities
y_prob = best_model.predict_proba(X_test)[:, 1]

# Use 0.5 threshold to prioritize fraud detection
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# 11. Evaluate the model
print("\nFINAL CLASSIFICATION REPORT")
print("=" * 45)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("ROC-AUC:", round(
    roc_auc_score(y_test, y_prob), 4
))

print("PR-AUC:", round(
    average_precision_score(y_test, y_prob), 4
))

# 12. Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()

print("\nFRAUD DETECTION SUMMARY")
print("Correctly detected fraud:", tp)
print("Missed fraud:", fn)
print("False fraud alarms:", fp)
print("Fraud recall:", round(tp / (tp + fn), 4))

# 13. Save confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Fraud"]
)

disp.plot(values_format="d")
plt.title("XGBoost Fraud Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 14. Save trained model and threshold
joblib.dump(
    {
        "model": best_model,
        "threshold": threshold
    },
    "fraud_detection_model.pkl"
)

print("\nModel saved: fraud_detection_model.pkl")
print("Plot saved: confusion_matrix.png")
