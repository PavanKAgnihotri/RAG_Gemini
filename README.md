# RAG_Gemini

This Project implements a simple Retrieval Augmented Generation (RAG) system. Here's a breakdown of each part:

1. RAG Genai 

  This part focuses on generating embeddings and building a vector database from your CSV data.
  
  1. Import Libraries: Imports necessary libraries for handling data, generating embeddings, and managing API keys.
     Configure API Key: Sets up your API key for accessing the Google Generative AI services. It assumes the API key is stored as an environment variable named API_KEY.
  
  2. Load the Generative Model: Loads the gemini-1.5-flash model, which is used for generating text responses later.
     row_to_string(row) Function: A helper function to convert each row of your dataframe into a string format. This string representation is what will be embedded.
  
  3. Read CSV file: Reads your specified CSV file into a pandas DataFrame.
     Select Required Columns: Filters the DataFrame to only include the columns you've listed in required_columns.
 
  4. Handle Missing Values: Fills any missing values in the selected columns with empty strings.
     Batch Processing: The code processes the data in batches to manage the rate limits of the embedding API.
 
  5. Generate Embeddings: For each row in a batch, it converts the row to a string using row_to_string and then generates an embedding for that string using genai.embed_content with the text-embedding-004 model.
  
  6. Handle Errors: Includes a basic error handling mechanism for embedding generation.
  
  7. Wait: Pauses for 60 seconds after processing each batch to comply with potential API rate limits.
  
  8. Normalize Embeddings: Normalizes the generated embeddings. This is a common practice in vector databases to improve similarity calculations.
  
  9. Create Vector Database: Combines the normalized embeddings with the original structured row strings into a list of dictionaries, forming your vector database.
  
  10. Convert NumPy Arrays: Converts any NumPy arrays within the vector database entries to lists so they can be easily saved to JSON.
  11. Save Vector Database: Saves the created vector database to a JSON file named vector_database.json.

2. RAG Retrieval (Code Cell fyEwCOAfYAL9)

This part focuses on retrieving relevant information from the vector database based on a user query and using that information to generate a response.

  1. Import Libraries: Imports necessary libraries for handling JSON, performing calculations (NumPy), calculating similarity, and interacting with the generative model.
  
  2. Configure API Key: Sets up the API key again for accessing the generative model.
  
  3. Load the Generative Model: Loads the gemini-1.5-flash model, which will be used to generate the final response.
  
  4. load_vector_database(json_file) Function: A function to load the vector database from the previously saved JSON file.
  
  5. embed_query(query, model) Function: Generates an embedding for the user's query using the text-embedding-004 model.
  
  6. retrieve_relevant_rows(query, vector_database, top_k) Function:
        Embeds the user query.
        Extracts the embeddings from the vector database.
        Calculates the cosine similarity between the query embedding and all the embeddings in the database. Cosine similarity measures the angle between two vectors and is a common way to determine how similar two embeddings are.
        Identifies the indices of the top_k rows with the highest similarity scores.
        Retrieves the corresponding original row data and their similarity scores.
        Returns the top-k relevant rows and their scores.
  
  7. generate_response(prompt, context, model) Function:
        Combines the user's prompt with the retrieved relevant context to create a comprehensive prompt for the generative model.
        Uses the gemini-1.5-flash model to generate a response based on the combined prompt.
        Returns the generated response.
  
  8. chatbot() Function:
        Initializes the chatbot interaction.
        Loads the vector database.
        Enters a loop to continuously accept user input.
        If the user types "quit" or "exit", the chatbot ends.
        For each user query, it retrieves the top 10 relevant rows from the vector database using retrieve_relevant_rows.
        Formats the retrieved data as context.
        Generates a response using generate_response, providing the user's query as the prompt and the retrieved data as the context.
        Prints the chatbot's response.

This RAG system allows you to query your data using natural language. The system first retrieves relevant information from your data using embeddings and then uses a generative model to synthesize a response based on that retrieved information.
