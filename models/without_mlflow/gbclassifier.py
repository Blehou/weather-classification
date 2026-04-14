import sys
import pandas as pd
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger
from utils.save import save_model

# Logger
gb_logger = setup_logger(
    "src/results/models/gradient_boosting/gradient_boosting.log",
    "gradient_boosting"
)

# Chargement données
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

# preprocessing
X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=False
)

# grille d'hyperparamètres
param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5, 7],
}

# GridSearch
grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# résultats CV
gb_logger.info(f"Best params: {grid.best_params_}")
gb_logger.info(f"Best CV score: {grid.best_score_:.4f}")

# validation
best_gb = grid.best_estimator_
y_pred = best_gb.predict(X_val)

gb_logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
gb_logger.info("Classification Report:")
gb_logger.info(f"\n{classification_report(y_val, y_pred)}")

save_model(
    best_gb,
    "src/results/models/gradient_boosting/gradient_boosting.pkl"
)

gb_logger.info(f"Model saved at: {'src/results/models/gradient_boosting/gradient_boosting.pkl'}")