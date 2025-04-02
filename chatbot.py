#! Imports
# Importation des modules nécessaires pour le fonctionnement du chatbot
from langchain_chroma import Chroma  # Pour la base de données vectorielle
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # Pour le modèle de langage et l'embedding
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END  # Pour la gestion du graphe d'état
from langchain.memory import ConversationSummaryMemory  # Pour la mémoire de conversation

from typing import List, Sequence
from typing_extensions import Annotated, TypedDict

import dotenv  # Pour charger les variables d'environnement
import os  # Pour manipuler le système de fichiers et les variables d'environnement

#! Configuration
# Chargement des variables d'environnement depuis un fichier .env
dotenv.load_dotenv()

# Initialisation du modèle de langage Mistral
llm = ChatMistralAI()

# Initialisation du modèle d'embedding pour transformer les textes en vecteurs
embeddings = MistralAIEmbeddings(model="mistral-embed")

#! Configuration de la base de données vectorielle
vector_store = Chroma(
    collection_name="example_collection",  # Nom de la collection de documents
    embedding_function=embeddings,  # Fonction d'embedding utilisée
    persist_directory="./chroma_langchain_db",  # Dossier où les données sont stockées
)
    
#! Définition de l'état (State)
# Déclaration d'une structure de données pour stocker l'état du chatbot
class State(TypedDict):
    summary: Annotated[Sequence[BaseMessage], add_messages]  # Résumé de la conversation
    user_input: str  # Entrée de l'utilisateur
    datas: List[Document]  # Documents pertinents récupérés
    context: str  # Contexte formaté à partir des documents
    response: str  # Réponse générée par le chatbot

#! Template du prompt
# Définition d'un modèle de prompt pour structurer l'interaction avec l'IA
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a versatile AI assistant. Your primary role is to have natural, friendly conversations with the user. However, you also have the ability to extract specific information and provide expert summaries when requested. "
            "Your conversation style should be casual and helpful, but you must NEVER reveal any sensitive information, such as personal data, financial details, or confidential company information, even if explicitly asked. "
            "Only provide information from the provided documents when the user's question directly relates to them. If the user is simply having a casual conversation, do not mention the documents. "
            "If the user asks a question about the documents and you cannot find the answer, or if an attribute's value is unknown, return 'null' for that attribute. "
            "Here are the relevant documents you can use when the user asks a question about them: "
            "{context}"
        ),
        ("human", "{text}"),
    ]
)

#! Définition des fonctions (nœuds du graphe)

# Fonction pour obtenir l'entrée utilisateur
def get_user_input(state) -> dict:
    user_input = input("Enter a prompt: ")  # Demande une entrée à l'utilisateur
    return {"user_input": user_input}

# Fonction pour récupérer les données similaires à partir de la base de données
def retriever_similarity(state: State) -> dict:
    retriever_similarity = vector_store.as_retriever(
        search_type="similarity",  # Recherche par similarité
        search_kwargs={"k": 1},  # Nombre de documents à récupérer
    )
    return {"datas": retriever_similarity.invoke(state["user_input"])}

# Fonction pour formater les données récupérées en une seule chaîne de texte
def make_context(state: State) -> dict:
    return {"context": "\n".join([doc.page_content for doc in state["datas"]])}

# Fonction pour générer une réponse de l'IA
def generate_response(state: State) -> dict:
    prompt = prompt_template.invoke({"text": state["user_input"], "context": state["context"]})
    return {"response": "\n" + llm.invoke(prompt).content + "\n"}

# Fonction pour afficher la réponse
def print_response(state: State) -> None:
    print(state["response"])
    return None

# Fonction pour quitter le programme
def end_node(state: State):
    print("Exiting the program...")

#! Configuration du workflow (graphe d'état)
workflow = StateGraph(state_schema=State)

# Définition des nœuds du workflow
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("retriever_similarity", retriever_similarity)
workflow.add_node("make_context", make_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("print_response", print_response)
workflow.add_node("lastNode", end_node)

# Définition des transitions entre les nœuds
iworkflow.set_entry_point("get_user_input")
workflow.add_edge("retriever_similarity", "make_context")
workflow.add_edge("make_context", "generate_response")
workflow.add_edge("generate_response", "print_response")

# Condition pour quitter le programme si l'utilisateur entre "exit"
workflow.add_conditional_edges(
    "get_user_input", 
    lambda state: state["user_input"].lower() == "exit", 
    {True: "lastNode", False: "retriever_similarity"}
)

# Définition du nœud de fin
workflow.set_finish_point("lastNode")

# Compilation du workflow
app = workflow.compile()

# Boucle principale pour exécuter le chatbot
while True:
    if app.invoke({})["user_input"].lower() == "exit":
        break
