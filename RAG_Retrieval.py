#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:19:26 2024

@author: pavan
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import os

# Configure API key
genai.configure(api_key = os.environ["API_KEY"])
m = genai.GenerativeModel("gemini-1.5-flash")

# Load vector database from JSON
def load_vector_database(json_file):
    with open(json_file, "r") as f:
        vector_database = json.load(f)
    return vector_database


def embed_query(query, model="models/text-embedding-004"):
    query_embedding = genai.embed_content(content=query, model=model)["embedding"]
    return np.array(query_embedding)

# Retrieve top-k relevant rows based on cosine similarity
def retrieve_relevant_rows(query, vector_database, top_k):
    # Embed query
    query_embedding = embed_query(query)
    
    # Extract embeddings and corresponding rows from the database
    embeddings = np.array([np.array(entry["embedding"]) for entry in vector_database])
    rows = [entry["data"] for entry in vector_database]
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    # Get top-k indices with the highest similarity scores
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Retrieve corresponding rows and similarity scores
    top_results = [(rows[i], similarities[i]) for i in top_indices]
    
    return top_results

def generate_response(prompt, context, model="gemini-1.5-flash"):
    # Combine prompt and context
    full_prompt = f"Context:\n{context}\n\nPrompt:\n{prompt}"
    
    response = m.generate_content(full_prompt)
    return response.text.strip()


def chatbot():
    print("Chatbot: Hello! How can I help you today?\nEnter ‘quit’ or ‘exit’ to end the session")
    vector_database = load_vector_database("vector_database.json")
    
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Chatbot: Goodbye!")
            break
        
        # Retrieve relevant rows
        retrieved_data = retrieve_relevant_rows(query, vector_database, 10)
        
        context = "\n".join([f"{i+1}. {row}" for i, (row, _) in enumerate(retrieved_data)])
        
        prompt = f"Based on the query '{query}', provide a detailed summary or answer."
        
        response = generate_response(prompt, context)
        
        print(f"Chatbot: {response}\n\n")

chatbot()

