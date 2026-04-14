# Image de base
FROM python:3.10-slim

# Eviter les fichiers .pyc + logs bufferisés
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dossier de travail
WORKDIR /app

# Copier requirements (depuis src/)
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier uniquement le code de l'app
COPY app/ .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer Streamlit
CMD ["streamlit", "run", "appli.py", "--server.port=8501", "--server.address=0.0.0.0"]