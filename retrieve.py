import os  # Importing the os module to interact with the operating system
import google.generativeai as genai  # Importing the google.generativeai module, which is used for generative AI functionalities provided by Google.
from ingest import load_docs, extract_and_split_docs, create_chroma_db, load_chroma_collection  

