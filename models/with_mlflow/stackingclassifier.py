import sys
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing

# =========================
# CONFIG MLflow
# =========================
EXPERIMENT_NAME = "weather_classification_stacking"
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOAD DATA
# =========================
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=True
)

# =========================
# BASE MODELS
# =========================
estimators = [
    ("svc", SVC(probability=True)),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=3100, max_depth=5, random_state=42))
]

final_estimator = LogisticRegression()

# =========================
# MLflow RUN
# =========================
with mlflow.start_run(run_name="StackingClassifier"):

    stack_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )

    # =========================
    # TRAINING
    # =========================
    stack_model.fit(X_train, y_train)

    # =========================
    # VALIDATION
    # =========================
    y_pred = stack_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)

    mlflow.log_metric("val_accuracy", val_accuracy)

    print(classification_report(y_val, y_pred))

    # =========================
    # LOG MODEL
    # =========================
    mlflow.sklearn.log_model(
        sk_model=stack_model,
        artifact_path="model"
    )

    # =========================
    # TAGS
    # =========================
    mlflow.set_tags({
        "project": "weather_prediction",
        "model": "StackingClassifier",
        "type": "ensemble"
    })