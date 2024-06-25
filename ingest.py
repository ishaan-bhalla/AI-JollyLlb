import os  # Standard library for interacting with the operating system
from dotenv import load_dotenv  # Dotenv package to load environment variables from a .env file
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PyPDFDirectoryLoader for loading PDF documents from a directory
from tqdm import tqdm  # tqdm for displaying a progress bar
from typing import List  # List for type annotations
import regex as re  # Regex for regular expression operations
import chromadb  # Chromadb for handling the ChromaDB database
from chromadb import Documents, EmbeddingFunction, Embeddings  # Classes for handling documents and embeddings in ChromaDB
import google.generativeai as genai  # Google generative AI package for embedding functions

# Load environment variables from a .env file
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")

# Function for Loading Documents using PyPDFDirectoryLoader
def load_docs(file_path):
    loader = PyPDFDirectoryLoader(file_path)
    # Loading Documents with a Progress Bar
    docs = []
    for doc in tqdm(loader.load(), desc="Loading documents", unit="doc"):
        docs.append(doc)
    return docs
documents = load_docs(file_path="data")

# Function for splitting data into chunks
def split_text(text: str) -> List[str]:
    split_text = re.split('\n \n', text)
    return [i for i in split_text if i != ""]

# Function for converting data into single string
def extract_and_split_docs(documents) -> List[str]:
    text = " ".join([doc.page_content for doc in documents])
    return split_text(text)
chunked_text = extract_and_split_docs(documents=documents)

# Defined the GeminiEmbeddingFunction class based on the updated Chroma interface
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = api_key
        if not gemini_api_key:
            raise ValueError("Google API Key not provided. Please provide GOOGLE_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Function to create or load ChromaDB collection
def create_chroma_db(documents: List, path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    collections = chroma_client.list_collections()
    if name in [collection.name for collection in collections]:
        db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    else:
        db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        for i, d in enumerate(documents):
            db.add(documents=d, ids=str(i))
    return db, name
db, name = create_chroma_db(documents=chunked_text, path="vectorstore", name="rag_ex")    

# Function to load an existing ChromaDB collection
def load_chroma_collection(path, name):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    return db
db = load_chroma_collection(path="vectorstore", name="rag_ex")
