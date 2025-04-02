#! Imports
# Importation des modules nécessaires pour le traitement des documents, la vectorisation et l'utilisation des modèles LLM
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import dotenv
import os

#! Configuration
# Chargement du fichier .env pour récupérer les variables d'environnement
dotenv.load_dotenv()

# Initialisation du modèle LLM (Large Language Model) pour le traitement du langage naturel
llm = ChatMistralAI()

# Initialisation du modèle d'embedding pour transformer le texte en vecteurs
embeddings = MistralAIEmbeddings(model="mistral-embed")

#! Database Configuration
# Configuration de la base de données vectorielle avec Chroma
vector_store = Chroma(
    collection_name="example_collection",  # Nom de la collection dans la base de données
    embedding_function=embeddings,        # Fonction d'embedding utilisée pour vectoriser les documents
    persist_directory="./chroma_langchain_db",  # Répertoire local pour sauvegarder les données
)

#! Get Files
# Chemin du fichier PDF à traiter
file_path = "./data/nke-10k-2023.pdf"

# Création d'une instance de PyPDFLoader pour charger le fichier PDF
loader = PyPDFLoader(file_path)

# Chargement des documents à partir du fichier PDF
docs = loader.load()

#! Splitte documents
# Création d'une instance de la classe RecursiveCharacterTextSplitter
# Cette classe permet de découper les documents en morceaux de texte (chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Taille maximale de chaque chunk (en caractères)
    chunk_overlap=200,  # Chevauchement entre les chunks (en caractères)
    add_start_index=True  # Ajout de l'index de début pour chaque chunk
)

# Découpage des documents en morceaux (chunks) à l'aide de la méthode split_documents()
all_splits = text_splitter.split_documents(docs)

#! Checks
# Transformation du contenu des deux premiers chunks en vecteurs
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

# Vérification que les deux vecteurs ont la même longueur
assert len(vector_1) == len(vector_2)

#! Indexation
# Ajout de tous les chunks vectorisés dans la base de données vectorielle
ids = vector_store.add_documents(documents=all_splits)