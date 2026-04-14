import os
import joblib
import pandas as pd
from pathlib import Path


def save_model(model: object, path: str):
    """
    sauvegarde un modèle dans un dossier donné.

    Args:
        model (object): Modèle entraîné à sauvegarder (ex: modèle scikit-learn)
        path (str): chemin complet du fichier de sauvegarde (incluant le nom et l'extension)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)



def save_dataframe(df: pd.DataFrame, chemin_dossier: str, nom_fichier: str, format_fichier: str="csv"):
    """
    Sauvegarde un DataFrame dans un dossier donné.

    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        chemin_dossier (str): chemin du dossier
        nom_fichier (str): nom sans extension
        format_fichier (str): format du fichier ("csv", "xlsx", "parquet", etc.)
    
    """

    # Créer le dossier s'il n'existe pas
    os.makedirs(chemin_dossier, exist_ok=True)

    # Construire le chemin complet
    chemin_complet = os.path.join(chemin_dossier, f"{nom_fichier}.{format_fichier}")

    # Sauvegarde selon le format
    if format_fichier == "csv":
        df.to_csv(chemin_complet, index=False)
    elif format_fichier == "xlsx":
        df.to_excel(chemin_complet, index=False)
    elif format_fichier == "parquet":
        df.to_parquet(chemin_complet, index=False)
    else:
        raise ValueError(f"Format non supporté : {format_fichier}")

    print(f"Fichier sauvegardé ici : {chemin_complet}")