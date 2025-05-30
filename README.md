# Chatbot RAG (Retrieval Augmented Generation)

Ce projet implémente un chatbot utilisant l'approche RAG (Retrieval Augmented Generation), qui enrichit les réponses du modèle avec des données contextuelles.

## Table des matières
1. [Installation](#installation)
2. [Prérequis](#prérequis)
3. [Configuration](#configuration)
4. [Utilisation](#utilisation)
5. [Structure du projet](#structure-du-projet)
6. [Dépannage](#dépannage)

---

## Installation

1. Clonez le dépôt Git :
   ```bash
   git clone https://github.com/Nadir-Bsd/RAG_LLM_Python.git
   ```

2. Accédez au répertoire du projet :
   ```bash
   cd RAG_LLM_Python
   ```

3. Installez les dépendances nécessaires :
   ```bash
   pip install -r requirements.txt
   ```

---

## Prérequis

Avant de commencer, assurez-vous d'avoir les éléments suivants :
- **Python** : Version 3.8 ou supérieure.
- **Dépendances** : Listées dans le fichier `requirements.txt`.
- **Clé API Mistral AI** : Obtenez-la [ici](https://console.mistral.ai/api-keys) (compte requis).
- **Clé API Langsmith** : Obtenez-la [ici](https://smith.langchain.com/) (compte requis).

---

## Configuration

1. Dupliquez le fichier `.env.example` et renommez-le `.env` :
   ```bash
   cp .env.example .env
   ```

2. Ouvrez le fichier `.env` et configurez vos clés API :
   ```
   MISTRAL_API_KEY=VotreCléAPI
   LANGSMITH_API_KEY=VotreCléAPI
   ```

---

## Utilisation

### Étape 1 : Préparer les documents
- Placez vos fichiers PDF dans le dossier `data/`. **Seuls les fichiers PDF sont acceptés.**

### Étape 2 : Créer la base de données
- Exécutez le script d'indexation pour générer la base de données vectorielle.

#### Sous Linux :
   ```bash
   python3 indexationPhase.py
   ```

#### Sous Windows :
   ```bash
   python indexationPhase.py
   ```

> **Note** : Ce processus peut prendre du temps en fonction de la taille des documents.

### Étape 3 : Interagir avec le chatbot
- Lancez le script du chatbot pour poser vos questions.

#### Sous Linux :
   ```bash
   python3 chatbot.py
   ```

#### Sous Windows :
   ```bash
   python chatbot.py
   ```

- Pour quitter le chatbot, tapez :
   ```bash
   exit
   ```

---

## Structure du projet

- `data/` : Contient les fichiers source (ajoutez ce dossier à `.gitignore` si les données sont sensibles).
- `chroma/` : Contient la base de données vectorielle générée.
- `indexationPhase.py` : Script pour créer la base de données vectorielle.
- `chatbot.py` : Script pour interagir avec le chatbot.

---

## Dépannage

### Problème : Les clés API ne fonctionnent pas
- Vérifiez que vos clés API sont correctement configurées dans le fichier `.env`.
- Assurez-vous que votre compte Mistral AI et Langsmith est actif.

### Problème : Erreur lors de l'installation des dépendances
- Assurez-vous que `pip` est à jour :
   ```bash
   pip install --upgrade pip
   ```

- Réessayez d'installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

### Problème : Le script d'indexation est lent
- Vérifiez que vos fichiers PDF ne sont pas trop volumineux.
- Assurez-vous que votre machine dispose de suffisamment de mémoire.

---

## Remerciements

Merci d'utiliser ce projet ! N'hésitez pas à contribuer ou à signaler des problèmes via le dépôt GitHub.