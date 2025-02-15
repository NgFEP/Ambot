import os
import json
import logging
import numpy as np
import re  # Import regex module for cleaning LLM response
from typing import List, Optional
import ollama
import faiss

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
EMBEDDINGS_DIR = "embeddings"  
DATA_FOLDER = "data"  
LLM_MODEL = "deepseek-r1:14b"  
EMBEDDING_MODEL = "nomic-embed-text"  

# DeepSeek-style system prompt
SYSTEM_PROMPT = """
[Instruction]
You are an intelligent assistant specialized in answering user questions based on the provided information.
Your responses must be:
1. **Fact-based** - Only use the retrieved context. If information is insufficient, respond with: "I do not have enough information to answer this."
2. **Concise and clear** - Avoid redundant explanations and keep responses direct.
3. **Structured** - Provide step-by-step answers where necessary.

[Context]
{context}

[User Query]
{query}

[Response]
"""

def load_embeddings(filename: str) -> Optional[List[np.ndarray]]:
    """
    Load embeddings from a JSON file.
    """
    base_name = os.path.basename(filename)
    path = os.path.join(EMBEDDINGS_DIR, f"{base_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [np.array(emb) for emb in json.load(f)]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0

def clean_response(response: str) -> str:
    """
    Remove unwanted tokens like <think>, </think>, or other similar artifacts.
    """
    response = re.sub(r"</?think>", "", response, flags=re.IGNORECASE).strip()  # Removes <think> and </think>
    return response

def retrieve_and_answer_query(user_query: str) -> None:
    """
    Retrieve relevant chunks based on a user query and generate an answer.
    """
    all_chunks, all_embeddings = [], []

    for file_name in os.listdir(DATA_FOLDER):
        if file_name.startswith("."):
            continue

        file_path = os.path.join(DATA_FOLDER, file_name)
        embeddings = load_embeddings(file_path)
        chunks_path = os.path.join(EMBEDDINGS_DIR, f"{os.path.basename(file_path)}_chunks.json")

        if embeddings and os.path.exists(chunks_path):
            with open(chunks_path, "r") as f:
                chunks = json.load(f)

            if len(chunks) == len(embeddings):
                all_chunks.extend(chunks)
                all_embeddings.extend(embeddings)
            else:
                logging.warning(f"Mismatch in chunks and embeddings for {file_name}. Skipping.")

    if not all_embeddings:
        logging.error("No valid embeddings found.")
        return

    # Generate query embedding
    query_embedding = np.array(ollama.embeddings(model=EMBEDDING_MODEL, prompt=user_query)["embedding"]).astype('float32')

    # Compute cosine similarities with all stored embeddings
    similarities = [cosine_similarity(query_embedding, emb) for emb in all_embeddings]

    # Get top 3 most similar chunks
    top_k = 3
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Sort in descending order

    # Retrieve relevant chunks
    relevant_chunks = [all_chunks[idx] for idx in top_indices]

    # Print similarity scores
    print("\n--- Top Retrieved Chunks & Cosine Similarities ---\n")
    for idx in top_indices:
        print(f"Similarity: {similarities[idx]:.4f}\nChunk: {all_chunks[idx][:300]}...\n")  # Preview first 300 chars
    print("-----------------------------------------------------\n")

    # Combine retrieved context
    context = "\n".join(relevant_chunks)

    # Format the DeepSeek-style prompt
    formatted_prompt = SYSTEM_PROMPT.format(context=context, query=user_query)

    # Generate response using the retrieved context
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": formatted_prompt},
            {"role": "user", "content": user_query}
        ],
        options={"temperature": 0}
    )

    # Extract the response text
    answer = response.get("message", {}).get("content", "No response received.")

    #  Remove "<think>", "</think>", and similar unwanted tags
    answer = clean_response(answer)

    print("\n--- LLM RESPONSE ---\n")
    print(answer)
    print("\n-----------------------------------------------------\n")

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ").strip()
        if user_query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        retrieve_and_answer_query(user_query)

