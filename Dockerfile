# Utilisez une image de base officielle Python 3.8
FROM python:3.9-slim

# Définir une variable d'environnement pour le répertoire de travail
WORKDIR /app

# Copier les fichiers requirements.txt dans le conteneur
COPY ./requirements.txt /app/requirements.txt

# Mettre à jour pip
RUN pip install --upgrade pip

# Installer les dépendances
RUN pip install -r requirements.txt

# Copier le reste des fichiers de l'application dans le conteneur
COPY . /app

# Exposer le port sur lequel l'application s'exécutera
EXPOSE 8000

# Définir la commande par défaut pour exécuter l'application
CMD ["cd", "Src/App"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]