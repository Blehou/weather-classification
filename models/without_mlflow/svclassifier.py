import sys
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger
from utils.save import save_model

svc_logger = setup_logger(
    "src/results/models/svc/svc.log",
    "svc"
)


path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

# preprocessing
X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(df=data, target_col="Weather Type")


param_grid = {
    "C": [0.5, 0.6, 0.7, 0.9, 1.0],
    "gamma": ["scale", "auto"]
}

grid = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

svc_logger.info(f"Best params: {grid.best_params_}")
svc_logger.info(f"Best CV score: {grid.best_score_:.4f}")

best_svc = grid.best_estimator_
y_pred = best_svc.predict(X_val)

svc_logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
svc_logger.info("Classification Report:")
svc_logger.info(f"\n{classification_report(y_val, y_pred)}")

save_model(
    best_svc,
    "src/results/models/svc/svc.pkl"
)

svc_logger.info(f"Model saved at: {'src/results/models/svc/svc.pkl'}")
