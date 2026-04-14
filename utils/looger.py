import logging
from pathlib import Path

def setup_logger(log_path: str, logger_name: str):
    """
    Crée un logger qui écrit dans un fichier + terminal.

    Args:
        log_path (str): chemin du fichier .log
        logger_name (str): nom du logger

    Returns:
        logging.Logger
    """

    # Créer le dossier si besoin
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Créer logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # éviter doublons si relancé
    if logger.handlers:
        return logger

    # format des logs
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # fichier
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # ajouter handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger