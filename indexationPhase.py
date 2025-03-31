
#! Imports
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import dotenv
import os

#! Configuration
# load the .env file
dotenv.load_dotenv()
# LLM model
llm = ChatMistralAI()
# embedding model
embeddings = MistralAIEmbeddings(model="mistral-embed")

#! Database Configuration
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)

#! Get Files
# Le chemin du fichier PDF
file_path = "./data/nke-10k-2023.pdf"
# cree l'instance de la class PyPDFLoader avec le chemin du fichier PDF en paramètre
loader = PyPDFLoader(file_path)
# La méthode load() permet de charger les documents
docs = loader.load()

#! Splitte documents
# instance de la class avec chunk_size=1000, chunk_overlap=200, add_start_index=True en paramètre
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
# la méthode split_documents() de la class permet de découper les documents en morceaux
all_splits = text_splitter.split_documents(docs)

#! Checks
# transforme les 2 premier chuck en vecteur
vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)
# check si les deux vecteurs ont la même longueur
assert len(vector_1) == len(vector_2)

#! Indexation
# prend tout les chucks les vectorises et les mets en DB vectorielle
ids = vector_store.add_documents(documents=all_splits)