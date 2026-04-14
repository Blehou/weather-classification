import joblib
import pandas as pd



# Charger les encodeurs sauvegardés
def load_encoders(ohe_path="artifacts/ohe.pkl", le_path="artifacts/label_encoder.pkl"):
    ohe = joblib.load(ohe_path)
    le = joblib.load(le_path)
    return ohe, le


def preprocess_input(input_data: dict, ohe, cat_cols: list):
    """
    Prétraitement identique au training (sans fit)

    Args:
        input_data (dict): données utilisateur
        ohe: OneHotEncoder déjà entraîné
        cat_cols (list): colonnes catégorielles

    Returns:
        pd.DataFrame: données encodées prêtes pour le modèle
    """

    # Convertir input en DataFrame
    df = pd.DataFrame([input_data])

    # =========================
    # ONE HOT ENCODING
    # =========================
    X_cat = ohe.transform(df[cat_cols])
    cat_names = ohe.get_feature_names_out(cat_cols)

    X_cat_df = pd.DataFrame(X_cat, columns=cat_names, index=df.index)

    # Supprimer colonnes catégorielles originales
    df = df.drop(columns=cat_cols)

    # Ajouter colonnes encodées
    df = pd.concat([df, X_cat_df], axis=1)

    return df

def align_columns(df, model_features):
    return df.reindex(columns=model_features, fill_value=0)