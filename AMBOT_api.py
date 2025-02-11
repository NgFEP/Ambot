import os
import json
import logging
from typing import List, Tuple

import numpy as np
from numpy.linalg import norm
from flask import Flask, request, jsonify, render_template
import ollama

# Import necessary functions and constants from your AMBOT.py file
from AMBOT import (
    generate_embeddings,
    find_most_similar,
    normalize_vector,
    SYSTEM_PROMPT,
    DATA_FOLDER
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)

# Preload embeddings and text chunks when the server starts.
# This calls your existing code to extract text, generate/load embeddings, etc.
global_chunks, global_embeddings = generate_embeddings(DATA_FOLDER)
logging.info("Embeddings and text chunks have been preloaded.")


def answer_query(user_query: str) -> str:
    """
    Process a user query by computing its embedding, retrieving relevant text chunks,
    and generating a response using the language model.
    """
    if not global_chunks or not global_embeddings:
        return "No chunks or embeddings loaded. Please run the embedding generation process."
    
    # Generate the query embedding using your existing model call.
    query_embedding_dict = ollama.embeddings(model="nomic-embed-text", prompt=user_query)
    query_embedding = normalize_vector(query_embedding_dict["embedding"])
    
    # Find the top 3 most similar text chunks.
    top_similar = find_most_similar(query_embedding, global_embeddings, top_k=5)
    #for score, idx in top_similar:
    #    logging.info(f"Score: {score:.4f}, Chunk Preview: {global_chunks[idx][:100]}...")
    
    # Retrieve the corresponding text chunks.
    relevant_context = [global_chunks[idx] for _, idx in top_similar if idx < len(global_chunks)]
    if not relevant_context:
        return "No relevant context found."
    
    context_snippets = "\n".join(relevant_context)
    
    # Generate the answer using the retrieved context.
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + context_snippets},
            {"role": "user", "content": f"Error: {user_query}"}
        ],
        options={"temperature": 0}
    )
    
    answer = response.get("message", {}).get("content", "No response received.")
    return answer


@app.route("/")
def home():
    """
    Render the homepage. Ensure you have an 'index.html' template in your templates folder.
    """
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query_endpoint():
    """
    Accept a query via POST, process it, and return an answer as JSON.
    Supports both JSON payloads (Content-Type: application/json) and form data.
    """
    if request.content_type and request.content_type.startswith("application/json"):
        data = request.get_json()
    else:
        data = request.form.to_dict()

    user_query = data.get("question", "").strip()
    if not user_query:
        return jsonify({"error": "No question provided."}), 400

    answer = answer_query(user_query)
    return jsonify({"question": user_query, "answer": answer})



if __name__ == "__main__":
    # Run the Flask app on localhost (default port 5000)
    app.run(debug=True)

