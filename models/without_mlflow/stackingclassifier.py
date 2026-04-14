import sys
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

sys.path.append(str(Path(__file__).resolve().parents[2]))

from preprocessing.pretraitement import further_preprocessing
from utils.looger import setup_logger

# Logger
stack_logger = setup_logger(
    "src/results/models/stacking/stacking.log",
    "stacking"
)

# Chargement données
path = Path("src/data/Preprocessed/weather_preprocessed.csv")
data = pd.read_csv(path)

# preprocessing
X_train, X_val, X_test, y_train, y_val, y_test = further_preprocessing(
    df=data,
    target_col="Weather Type",
    scale=True 
)

# --- modèles de base ---
estimators = [
    ("svc", SVC(probability=True)),
    ("rf", RandomForestClassifier(n_estimators=300, random_state=42)),
    ("gb", GradientBoostingClassifier(n_estimators=3100, max_depth=5, random_state=42))
]

# --- meta model ---
final_estimator = LogisticRegression()

# --- stacking ---
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=5,
    n_jobs=-1
)

# entraînement
stack_model.fit(X_train, y_train)

# validation
y_pred = stack_model.predict(X_val)

# logs
stack_logger.info("Stacking model trained")
stack_logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
stack_logger.info("Classification Report:")
stack_logger.info(f"\n{classification_report(y_val, y_pred)}")