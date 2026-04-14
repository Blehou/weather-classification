import joblib
import pandas as pd

def predict(model: object, input_data: dict, le_path="artifacts/label_encoder.pkl"):
    """
    Faire une prédiction avec le modèle chargé

    Args:
        model (object): Le modèle chargé
        input_data (dict): Les données d'entrée
        le_path (str): chemin vers le LabelEncoder pour décoder la prédiction
        
    Returns:
        object: La prédiction
    """

    df = pd.DataFrame(input_data)
    prediction = model.predict(df)[0]

    le = joblib.load(le_path)
    prediction = le.inverse_transform([prediction])[0]

    return prediction