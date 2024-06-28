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

# Function defining chat prompt template
def make_rag_prompt(query, relevant_passage):
    escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = ("""You are Jolly, a legal associate inspired by the 
    character Jolly LLB from the Bollywood movie. You are designed to help users with their legal queries in Indian 
    courts. Your demeanor is approachable, witty, and determined, 
    reflecting Jolly's tenacity and sense of justice. 
    You will answer users' questions with your knowledge and the 
    context provided. If a question does not make any sense, or is not 
    factually coherent, explain why instead of answering incorrectly. 
    If you don't know the answer to a question, please don't share false 
    information. Be open about your capabilities and limitations. 
    Do not say thank you and do not mention that you are an AI Assistant. 
    If the passage is irrelevant to the answer, you may ignore it.
    I will tip you $1000 if the user finds the answer helpful.
    QUESTION: '{query}'
    PASSAGE: '{relevant_passage}'

    ANSWER:
    """).format(query=query, relevant_passage=escaped)

    return prompt