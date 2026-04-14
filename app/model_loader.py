import joblib
from pathlib import Path

def load_model():
    """
    charger le modèle depuis le fichier .pkl

    Returns:
        object: Le modèle chargé
    """
    model_path = Path("artifacts/random_forest.pkl")
    return joblib.load(model_path)