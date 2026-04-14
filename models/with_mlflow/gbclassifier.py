import sys
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger
from utils.save import save_model

# =========================
# CONFIG
# =========================
EXPERIMENT_NAME = "weather_classification_gb"

mlflow.set_experiment(EXPERIMENT_NAME)

# =========================
# LOGGER
# =========================
gb_logger = setup_logger(
    "src/results/models/gradient_boosting/gradient_boosting.log",
    "gradient_boosting"
)

# =========================
# LOAD DATA
# =========================
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=False
)

# =========================
# PARAM GRID
# =========================
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
}

# =========================
# MLFLOW RUN
# =========================
with mlflow.start_run(run_name="GradientBoosting_GridSearch"):

    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    # =========================
    # BEST PARAMS
    # =========================
    best_params = grid.best_params_
    best_cv_score = grid.best_score_

    gb_logger.info(f"Best params: {best_params}")
    gb_logger.info(f"Best CV score: {best_cv_score:.4f}")

    mlflow.log_params(best_params)
    mlflow.log_metric("cv_accuracy", best_cv_score)

    # =========================
    # VALIDATION
    # =========================
    best_gb = grid.best_estimator_
    y_pred = best_gb.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)

    gb_logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    gb_logger.info("Classification Report:")
    gb_logger.info(f"\n{classification_report(y_val, y_pred)}")

    mlflow.log_metric("val_accuracy", val_accuracy)

    # =========================
    # SAVE MODEL (LOCAL + MLFLOW)
    # =========================
    save_path = "src/results/models/gradient_boosting/gradient_boosting.pkl"
    save_model(best_gb, save_path)

    gb_logger.info(f"Model saved at: {save_path}")

    # MLflow model logging
    mlflow.sklearn.log_model(
        sk_model=best_gb,
        artifact_path="model"
    )

    # =========================
    # TAGS
    # =========================
    mlflow.set_tags({
        "project": "weather_prediction",
        "model": "GradientBoosting",
        "stage": "training"
    })