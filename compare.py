import sys
import joblib
import mlflow
import pandas as pd
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[1]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger

# ==============
# Logger
# ==============
compare_logger = setup_logger(
    "src/results/models/compare.log",
    "compare"
)

compare_logger.info("===== START MODEL COMPARISON =====")

# ==============
# MLflow config
# ==============
mlflow.set_experiment("weather_model_comparison")

# ===============
# DATA
# ===============
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

# ===============
# PREPROCESSING
# ===============
X_train_ns, X_val_ns, X_test_ns, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=False
)

X_train_s, X_val_s, X_test_s, _, _, _ = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=True
)

# ==============
# LOAD MODELS
# ==============
svc_model = joblib.load("src/results/models/svc/svc.pkl")
rf_model = joblib.load("src/results/models/random_forest/random_forest.pkl")
gb_model = joblib.load("src/results/models/gradient_boosting/gradient_boosting.pkl")

models = {
    "SVC": (svc_model, X_test_s),
    "RandomForest": (rf_model, X_test_ns),
    "GradientBoosting": (gb_model, X_test_ns)
}

# ==============================
# COMPARISON
# ==============================
results = {}

for name, (model, X_test_used) in models.items():

    compare_logger.info(f"===== {name} =====")

    with mlflow.start_run(run_name=f"compare_{name}"):

        y_pred = model.predict(X_test_used)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        results[name] = acc

        # MLflow tracking (minimal mais utile)
        mlflow.log_metric("test_accuracy", acc)
        mlflow.set_tag("model", name)

        compare_logger.info(f"Test Accuracy: {acc:.4f}")
        compare_logger.info("Classification Report:")
        compare_logger.info(f"\n{report}")

# ==============================
# BEST MODEL
# ==============================
best_model = max(results, key=results.get)

compare_logger.info("===== BEST MODEL =====")
compare_logger.info(f"Best model: {best_model}")
compare_logger.info(f"Best accuracy: {results[best_model]:.4f}")

with mlflow.start_run(run_name="best_model_summary"):
    mlflow.log_metric("best_accuracy", results[best_model])
    mlflow.set_tag("best_model", best_model)

compare_logger.info("===== END MODEL COMPARISON =====")