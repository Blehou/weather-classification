import sys
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.save import save_model

# =========================
# CONFIG MLflow
# =========================
EXPERIMENT_NAME = "weather_classification_rf"
mlflow.set_experiment(EXPERIMENT_NAME)

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
    "max_features": ["sqrt", "log2"],
}

# =========================
# MLflow RUN
# =========================
with mlflow.start_run(run_name="RandomForest_GridSearch"):

    grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
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
    best_rf = grid.best_estimator_
    y_pred = best_rf.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_pred)

    mlflow.log_metric("val_accuracy", val_accuracy)

    # (optionnel mais utile debug)
    print(classification_report(y_val, y_pred))

    # =========================
    # SAVE MODEL LOCAL + MLFLOW
    # =========================
    save_model(
        best_rf,
        "src/results/models/random_forest/random_forest.pkl"
    )

    mlflow.sklearn.log_model(
        sk_model=best_rf,
        artifact_path="model"
    )

    # =========================
    # TAGS PROJET
    # =========================
    mlflow.set_tags({
        "project": "weather_prediction",
        "model": "RandomForest",
        "stage": "training"
    })