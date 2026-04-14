import os
import sys
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.save import save_dataframe
from utils.looger import setup_logger

# Logger pour le prétraitement
preprocess_logger = setup_logger(
    "src/results/preprocessing/preprocessing.log",
    "preprocessing"
)


def encode_dataframe(df: pd.DataFrame, cat_cols: list, target_col: str, verbose: bool=True, ohe_path: str="src/results/preprocessing/artifacts/ohe.pkl", le_path: str="src/results/preprocessing/artifacts/label_encoder.pkl") -> pd.DataFrame:
    """ Encode les variables categorielles et la target

    Args:
        df (pd.DataFrame): Le DataFrame à encoder
        cat_cols (list): La liste des colonnes categorielles à encoder
        target_col (str): Le nom de la colonne cible à encoder
        verbose (bool): Si True, affiche les messages de progression
        ohe_path (str): Le chemin pour sauvegarder l'encodeur OneHotEncoder
        le_path (str): Le chemin pour sauvegarder l'encodeur LabelEncoder

    Returns:
        pd.DataFrame: Le DataFrame encode
    """
    
    df_out = df.copy()
    
    # --- One Hot Encoding sur les variables explicatives ---
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    
    X_cat = ohe.fit_transform(df_out[cat_cols])
    cat_names = ohe.get_feature_names_out(cat_cols)
    
    X_cat_df = pd.DataFrame(X_cat, columns=cat_names, index=df_out.index)

    if verbose:
        preprocess_logger.info(f"Encodage One Hot terminee pour les colonnes : {cat_cols}")
    
    # Supprimer anciennes colonnes categorielles
    df_out = df_out.drop(columns=cat_cols)
    
    # Ajouter nouvelles colonnes
    df_out = pd.concat([df_out, X_cat_df], axis=1)
    
    # --- Encodage de la target ---
    le = LabelEncoder()
    df_out[target_col] = le.fit_transform(df_out[target_col])

    # Reorganiser les colonnes du dataframe
    cols = [col for col in df_out.columns if col != target_col] + [target_col]
    df_out = df_out[cols]

    if os.path.dirname(ohe_path):
        os.makedirs(os.path.dirname(ohe_path), exist_ok=True)

    if os.path.dirname(le_path):
        os.makedirs(os.path.dirname(le_path), exist_ok=True)

    joblib.dump(ohe, ohe_path)
    joblib.dump(le, le_path)

    if verbose:
        preprocess_logger.info(f"Encodage de la target '{target_col}' termine.")
    
    return df_out

def remove_anomalies(df: pd.DataFrame, thresholds: dict, verbose: bool=True) -> pd.DataFrame:
    """
    Supprime les anomalies selon des seuils définis

    Args:
        df (pd.DataFrame): Le DataFrame à nettoyer
        thresholds (dict): Un dictionnaire des seuils pour chaque colonne
        verbose (bool): Si True, affiche les messages de progression

    Returns:
        pd.DataFrame: Le DataFrame nettoyé
    
    Example:
    >>> thresholds = {
    ...     "Humidity": (0, 100),
    ...     "Precipitation (%)": (0, 100),
    ...     "Temperature": (-50, 60),
    ...     "Atmospheric Pressure": (900, 1100)
    ... }
    >>> df_clean = remove_outliers(df, thresholds)
    """
    
    df_clean = df.copy()
    
    for col, (min_val, max_val) in thresholds.items():
        df_clean = df_clean[
            (df_clean[col] >= min_val) & (df_clean[col] <= max_val)
        ]
    
    if verbose:
        preprocess_logger.info(f"Suppression des anomalies terminee selon les seuils : {thresholds}")
    
    return df_clean

def cap_outliers_iqr(df: pd.DataFrame, num_cols: list, verbose: bool=True) -> pd.DataFrame:
    """
    Limite les valeurs aberrantes (outliers) en les ramenant dans un intervalle
    défini par la méthode IQR (Interquartile Range).

    Les valeurs inférieures à Q1 - 1.5 * IQR sont remplacées par cette borne,
    et les valeurs supérieures à Q3 + 1.5 * IQR sont remplacées par cette borne.

    Args:
        df (pd.DataFrame): Le DataFrame à nettoyer
        num_cols (list): La liste des colonnes numériques à traiter
        verbose (bool): Si True, affiche les messages de progression

    Returns:
        pd.DataFrame: Le DataFrame nettoyé
    """
    
    df_out = df.copy()
    
    for col in num_cols:
        
        # Q1 : 25% des données
        Q1 = df_out[col].quantile(0.25)
        
        # Q3 : 75% des données
        Q3 = df_out[col].quantile(0.75)
        
        # IQR : dispersion centrale
        IQR = Q3 - Q1
        
        # bornes
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # clipping
        df_out[col] = df_out[col].clip(lower, upper)
    
    if verbose:
        preprocess_logger.info(f"Limitation des outliers terminee pour les colonnes : {num_cols}")
    
    return df_out

def further_preprocessing(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    scale: bool = True,
    random_state: int = 42
):
    """
    Split train/val/test + normalisation optionnelle.

    Args:
        df (pd.DataFrame): dataset complet
        target_col (str): nom de la variable cible
        test_size (float): proportion du test set
        val_size (float): proportion du validation set (dans le train initial)
        scale (bool): appliquer StandardScaler ou non
        random_state (int): reproductibilité

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    # --- séparation features / target ---
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- split train + test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y # stratify pour garder la même répartition de la target dans les splits
    )

    # --- split train + validation ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train
    )

    # --- normalisation (fit uniquement sur train) ---
    if scale:
        scaler = StandardScaler()

        num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns

        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    return X_train, X_val, X_test, y_train, y_val, y_test



if __name__ == "__main__":
    preprocess_logger.info("Debut preprocessing")
    
    # Chargement des données
    path = Path("src/data/Input/weather_classification_data.csv")
    data = pd.read_csv(path)

    # 1. Anomalies
    thresholds = {
        "Humidity": (0, 100),
        "Precipitation (%)": (0, 100),
        "Temperature": (-50, 60),
        "Atmospheric Pressure": (900, 1100)
    }

    data_clean = remove_anomalies(data, thresholds)

    # 2. Outliers
    num_cols = [
        "Temperature", "Humidity", "Wind Speed",
        "Precipitation (%)", "Atmospheric Pressure",
        "UV Index", "Visibility (km)"
    ]

    data_clean = cap_outliers_iqr(data_clean, num_cols)

    # 3. Encodage
    cat_cols = ["Cloud Cover", "Season", "Location"]

    data_final = encode_dataframe(data_clean, cat_cols, "Weather Type")

    # Sauvegarde du DataFrame prétraité
    save_dataframe(data_final, "src/data/Preprocessed", "weather_preprocessed", "csv")