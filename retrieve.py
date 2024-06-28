import os  # Importing the os module to interact with the operating system
import google.generativeai as genai  # Importing the google.generativeai module, which is used for generative AI functionalities provided by Google.
from ingest import load_docs, extract_and_split_docs, create_chroma_db, load_chroma_collection  

# This function retrieves the most relevant passage from the database based on the given query.
def get_relevant_passage(query, db, n_results):
    try:
        results = db.query(query_texts=[query], n_results=n_results)
        if 'documents' in results and results['documents']:
            passage = results['documents'][0]
            print(passage)
            return passage
        else:
            raise ValueError("No documents found for the query.")
    except Exception as e:
        print(f"Error retrieving passage: {e}")
        return None