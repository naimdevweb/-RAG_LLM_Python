
#! Imports
from langchain_chroma import Chroma

from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.messages import BaseMessage

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END

from langchain.memory import ConversationSummaryMemory

from typing import List, Sequence
from typing_extensions import Annotated, TypedDict


import dotenv
import os

#! Configuration
dotenv.load_dotenv()
llm = ChatMistralAI()
embeddings = MistralAIEmbeddings(model="mistral-embed")

#! Database Configuration
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
    
#! Define State
class State(TypedDict):
    summary: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str
    datas: List[Document]
    context: str
    response: str

#! Template
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

#! Functions Nodes
# get user input
def get_user_input(state) -> dict:
    user_input = input("Enter a prompt: ")
    return {"user_input": user_input}

# get datas from the database for making context
def retriever_similarity(state: State) -> dict:
    # transforme le prompt(query) en vecteur et fait une recherche de similarité
    retriever_similarity = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    # fait appel à l'object retriever_similarity et utlisant la méthode invoke pour obtenir les résultats
    return {"datas": retriever_similarity.invoke(state["user_input"])}

# format datas into a string
def make_context(state: State) -> dict:
    return {"context": "\n".join([doc.page_content for doc in state["datas"]])}

# generate response
def generate_response(state: State) -> dict:
    # put the context and the userPrompt in the prompt_Template 
    prompt = prompt_template.invoke({"text": state["user_input"], "context": state["context"]})
    
    # send prompt to the LLM
    return {"response": "\n" + llm.invoke(prompt).content + "\n"}

# print response
def print_response(state: State) -> None:
    print(state["response"])
    return None

# exit the program
def end_node(state: State):
    print("Exiting the program...")


#! SET UP WORKFLOW
workflow = StateGraph(state_schema=State)

# Define nodes in the workflow
workflow.add_node("get_user_input", get_user_input)
workflow.add_node("retriever_similarity", retriever_similarity)
workflow.add_node("make_context", make_context)
workflow.add_node("generate_response", generate_response)
workflow.add_node("print_response", print_response)
workflow.add_node("lastNode", end_node)

# Define edges in the workflow
workflow.set_entry_point("get_user_input")
workflow.add_edge("retriever_similarity", "make_context")
workflow.add_edge("make_context", "generate_response")
workflow.add_edge("generate_response", "print_response")
# check what get_user_input get and make boolean condition
workflow.add_conditional_edges("get_user_input", lambda state: state["user_input"].lower() == "exit", {True: "lastNode", False: "retriever_similarity"})
# exit Node
workflow.set_finish_point("lastNode")

# compile the workflow
app = workflow.compile()

while True:
          
    if app.invoke({})["user_input"].lower() == "exit":
        break