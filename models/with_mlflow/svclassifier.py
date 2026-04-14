import sys
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing

# =========================
# CONFIG MLflow
# =========================
EXPERIMENT_NAME = "weather_classification_svc"
mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOAD DATA
# =========================
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type"
)

# =========================
# PARAM GRID
# =========================
param_grid = {
    "C": [0.5, 0.6, 0.7, 0.9, 1.0],
    "gamma": ["scale", "auto"]
}

# =========================
# MLflow RUN
# =========================
with mlflow.start_run(run_name="SVC_GridSearch"):

    grid = GridSearchCV(
        SVC(),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # =========================
    # BEST RESULTS
    # =========================
    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("cv_accuracy", best_cv_score)

    # =========================
    # VALIDATION
    # =========================
    best_svc = grid.best_estimator_
    y_pred = best_svc.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)

    mlflow.log_metric("val_accuracy", val_accuracy)

    print(classification_report(y_val, y_pred))

    # =========================
    # LOG MODEL
    # =========================
    mlflow.sklearn.log_model(
        sk_model=best_svc,
        artifact_path="model"
    )

    # =========================
    # TAGS
    # =========================
    mlflow.set_tags({
        "project": "weather_prediction",
        "model": "SVC",
        "stage": "training"
    })