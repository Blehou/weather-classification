import sys
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger
from utils.save import save_model

# Logger
rf_logger = setup_logger(
    "src/results/models/random_forest/random_forest.log",
    "random_forest"
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
    "max_features": ["sqrt", "log2"],
}

# GridSearch
grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# résultats CV
rf_logger.info(f"Best params: {grid.best_params_}")
rf_logger.info(f"Best CV score: {grid.best_score_:.4f}")

# validation
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_val)

rf_logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
rf_logger.info("Classification Report:")
rf_logger.info(f"\n{classification_report(y_val, y_pred)}")

save_model(
    best_rf,
    "src/results/models/random_forest/random_forest.pkl"
)

rf_logger.info(f"Model saved at: {'src/results/models/random_forest/random_forest.pkl'}")