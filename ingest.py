import os  # Standard library for interacting with the operating system
from dotenv import load_dotenv  # Dotenv package to load environment variables from a .env file
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PyPDFDirectoryLoader for loading PDF documents from a directory
from tqdm import tqdm  # tqdm for displaying a progress bar
from typing import List  # List for type annotations
import regex as re  # Regex for regular expression operations
import chromadb  # Chromadb for handling the ChromaDB database
from chromadb import Documents, EmbeddingFunction, Embeddings  # Classes for handling documents and embeddings in ChromaDB
import google.generativeai as genai  # Google generative AI package for embedding functions

