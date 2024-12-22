#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 17:41:07 2024

@author: pavan
"""

import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import json, time

# Configure API Key
genai.configure(api_key = os.environ["API_KEY"])

# Load the Generative Model
model = genai.GenerativeModel("gemini-1.5-flash")

# Function to structure row as a dictionary string
def row_to_string(row):
    return ', '.join([f"'{col}': {value}" for col, value in row.items()])

# Read CSV file
csv_file = pd.read_csv("data_scientist_salaries.csv")  # Replace with your CSV file path
required_columns = ['Hobby', 'OpenSource', 'Country', 'Student', 'Employment', 'FormalEducation',
                    'UndergradMajor', 'CompanySize', 'DevType', 'YearsCoding', 'Salary', 'SalaryType', 'ConvertedSalary']

data = csv_file[required_columns]
data.fillna("", inplace=True)

batch_size = 15  # Number of requests per minute
embeddings = []
structured_rows = []

# Process data in batches
for i in range(0, len(data), batch_size):
    batch = data.iloc[i:i + batch_size]
    
    for _, row in batch.iterrows():
        row_string = row_to_string(row)
        structured_rows.append(row_string)
        
        try:
            # Generate embedding
            embedding = genai.embed_content(content=row_string, model="models/text-embedding-004")
            embeddings.append(embedding["embedding"])
        except Exception as e:
            print(f"Error for row: {row_string}, {e}")
            embeddings.append(None)
    
    # Wait for 1 minute after processing each batch
    print(f"Completed batch {i // batch_size + 1}. Waiting for the next batch...")
    time.sleep(60)
# Normalize embeddings
valid_embeddings = [embed for embed in embeddings if embed is not None]
normalized_embeddings = normalize(valid_embeddings)

# Create a vector database
vector_database = [{"embedding": norm_embed, "data": structured_row} 
                    for norm_embed, structured_row in zip(normalized_embeddings, structured_rows)]


for entry in vector_database:
    if isinstance(entry['embedding'], np.ndarray):
        entry['embedding'] = entry['embedding'].tolist()
# Save vector database to a JSON file
with open("vector_database.json", "w") as f:
    json.dump(vector_database, f)

print("Vector database created successfully!")

