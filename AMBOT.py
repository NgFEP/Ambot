import os
import json
import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from numpy.linalg import norm
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define chunk sizes for PDFs and HTML
PDF_CHUNK_SIZE = 500
HTML_CHUNK_SIZE = 500
EMBEDDINGS_DIR = "embeddings"
DATA_FOLDER = "data"
SYSTEM_PROMPT = (
    "You are a helpful assistant who answers questions based on the provided context. "
    "Be concise, and if unsure, say you don't know.\nContext:\n"
)


########################################################################
# 1. HELPER FUNCTIONS
########################################################################

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector (np.ndarray): The vector to normalize.
    
    Returns:
        np.ndarray: The normalized vector.
    """
    norm_value = np.linalg.norm(vector)
    return vector if norm_value == 0 else vector / norm_value


def save_embeddings(filename: str, embeddings: List[np.ndarray]) -> None:
    """
    Save embeddings to a JSON file under the EMBEDDINGS_DIR directory.
    
    Args:
        filename (str): The source file name.
        embeddings (List[np.ndarray]): List of embeddings.
    """
    base_name = os.path.basename(filename)
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)
    
    serializable_embeddings = [emb.tolist() for emb in embeddings]
    output_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}.json")
    with open(output_path, "w") as f:
        json.dump(serializable_embeddings, f)
    logging.info(f"Embeddings saved to {output_path}")


def load_embeddings(filename: str) -> Optional[List[np.ndarray]]:
    """
    Load embeddings from a JSON file.
    
    Args:
        filename (str): The source file name.
    
    Returns:
        Optional[List[np.ndarray]]: A list of embeddings or None if not found.
    """
    base_name = os.path.basename(filename)
    embeddings_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}.json")
    
    if not os.path.exists(embeddings_path):
        return None

    with open(embeddings_path, "r") as f:
        embeddings_list = json.load(f)
    
    return [np.array(emb) for emb in embeddings_list]


def find_most_similar(query_embedding: np.ndarray, embedding_list: List[np.ndarray], top_k: int = 5) -> List[Tuple[float, int]]:
    """
    Find the most similar chunks using cosine similarity.
    
    Args:
        query_embedding (np.ndarray): The embedding of the query.
        embedding_list (List[np.ndarray]): A list of chunk embeddings.
        top_k (int): Number of top similar items to return.
    
    Returns:
        List[Tuple[float, int]]: List of tuples (similarity_score, index).
    """
    query_norm = norm(query_embedding)
    similarities = [
        (np.dot(query_embedding, emb) / (query_norm * norm(emb)), idx)
        for idx, emb in enumerate(embedding_list)
    ]
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]


########################################################################
# 2. TEXT EXTRACTION & CHUNKING FUNCTIONS
########################################################################

def parse_pdf_file(filename: str, chunk_size: int = PDF_CHUNK_SIZE) -> List[str]:
    """
    Extract and chunk text from a PDF file.
    
    Args:
        filename (str): Path to the PDF file.
        chunk_size (int): Number of words per chunk.
    
    Returns:
        List[str]: List of text chunks.
    """
    reader = PdfReader(filename)
    text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return [chunk for chunk in chunks if chunk.strip()]


def parse_html_links(filename: str, chunk_size: int = HTML_CHUNK_SIZE) -> List[str]:
    """
    Extract and chunk text from HTML links provided in a file.
    
    Args:
        filename (str): Path to the file containing URLs.
        chunk_size (int): Number of words per chunk.
    
    Returns:
        List[str]: List of text chunks extracted from HTML pages.
    """
    with open(filename, encoding="utf-8-sig") as f:
        links = f.read().strip().splitlines()

    all_chunks: List[str] = []
    for link in links:
        try:
            response = requests.get(link, timeout=10)
            if response.status_code != 200:
                logging.warning(f"Non-200 status code for URL: {link}")
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            words = page_text.split()
            chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
            all_chunks.extend(chunks)

        except Exception as e:
            logging.error(f"Error processing link {link}: {e}")

    logging.info(f"Total HTML chunks extracted: {len(all_chunks)}")
    return all_chunks


########################################################################
# 3. EMBEDDING GENERATION
########################################################################

def get_embeddings(filename: str, modelname: str, chunks: List[str]) -> List[np.ndarray]:
    """
    Generate and save embeddings if they don't already exist.
    
    Args:
        filename (str): The source file name.
        modelname (str): The model name to use for embedding generation.
        chunks (List[str]): List of text chunks.
    
    Returns:
        List[np.ndarray]: List of embeddings.
    """
    embeddings = load_embeddings(filename)
    if embeddings is not None:
        return embeddings
    
    embeddings = [
        normalize_vector(ollama.embeddings(model=modelname, prompt=chunk)["embedding"])
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings


def generate_embeddings(data_folder: str) -> Tuple[List[str], List[np.ndarray]]:
    """
    Generate embeddings for all supported files in the data folder.
    
    Args:
        data_folder (str): Directory containing data files.
    
    Returns:
        Tuple[List[str], List[np.ndarray]]: Combined text chunks and corresponding embeddings.
    """
    combined_chunks: List[str] = []
    combined_embeddings: List[np.ndarray] = []
    
    pdf_chunk_count = 0
    html_chunk_count = 0

    for file_name in os.listdir(data_folder):
        if file_name.startswith("."):
            continue

        file_path = os.path.join(data_folder, file_name)
        if file_name.endswith(".pdf"):
            chunks = parse_pdf_file(file_path, chunk_size=PDF_CHUNK_SIZE)
            pdf_chunk_count += len(chunks)
        elif file_name == "amber_archive_urls.txt":
            chunks = parse_html_links(file_path, chunk_size=HTML_CHUNK_SIZE)
            html_chunk_count += len(chunks)
        else:
            logging.info(f"Skipping unsupported file format: {file_name}")
            continue

        embeddings = get_embeddings(file_path, "nomic-embed-text", chunks)
        combined_chunks.extend(chunks)
        combined_embeddings.extend(embeddings)

    logging.info(f"PDF chunks extracted: {pdf_chunk_count}")
    logging.info(f"HTML chunks extracted: {html_chunk_count}")
    logging.info(f"Total chunks extracted: {pdf_chunk_count + html_chunk_count}")

    return combined_chunks, combined_embeddings


########################################################################
# 4. QUERY RETRIEVAL & RESPONSE GENERATION
########################################################################

def retrieve_and_answer_query(data_folder: str) -> None:
    """
    Retrieve relevant chunks based on a user query and generate an answer.
    
    Args:
        data_folder (str): Directory containing data files.
    """
    combined_chunks: List[str] = []
    combined_embeddings: List[np.ndarray] = []

    # Load embeddings and text chunks from files
    for file_name in os.listdir(data_folder):
        if file_name.startswith("."):
            continue

        file_path = os.path.join(data_folder, file_name)
        if file_name.endswith(".pdf"):
            chunks = parse_pdf_file(file_path, chunk_size=PDF_CHUNK_SIZE)
        elif file_name == "amber_archive_urls.txt":
            chunks = parse_html_links(file_path, chunk_size=HTML_CHUNK_SIZE)
        else:
            logging.info(f"Skipping unsupported file format: {file_name}")
            continue

        embeddings = load_embeddings(file_path)
        if embeddings is None:
            logging.warning(f"No existing embeddings found for '{file_name}'. Run embedding generation first.")
            continue

        if len(chunks) != len(embeddings):
            logging.warning(
                f"Mismatch in chunks ({len(chunks)}) and embeddings ({len(embeddings)}) for {file_name}"
            )
            continue

        combined_chunks.extend(chunks)
        combined_embeddings.extend(embeddings)

    logging.info(f"Total chunks loaded: {len(combined_chunks)}")
    logging.info(f"Total embeddings loaded: {len(combined_embeddings)}")

    if not combined_chunks or not combined_embeddings:
        logging.error("No chunks or embeddings found. Ensure embeddings were generated.")
        return

    # User query input
    user_query = input("Describe the error -> ").strip()
    if not user_query:
        logging.error("No error provided. Exiting.")
        return

    # Generate the query embedding
    query_embedding_dict = ollama.embeddings(model="nomic-embed-text", prompt=user_query)
    query_embedding = normalize_vector(query_embedding_dict["embedding"])
    
    # Find the top-3 most relevant chunks
    top_similar = find_most_similar(query_embedding, combined_embeddings, top_k=3)
    for score, idx in top_similar:
        if idx < len(combined_chunks):
            logging.info(f"Score: {score:.4f}, Chunk Preview: {combined_chunks[idx][:100]}...")
        else:
            logging.warning(f"Index {idx} out of range for combined_chunks.")

    # Prepare context from the relevant chunks
    relevant_context = [combined_chunks[idx] for _, idx in top_similar if idx < len(combined_chunks)]
    if not relevant_context:
        logging.error("No relevant context found.")
        return

    context_snippets = "\n".join(relevant_context)

    # Generate response using the retrieved context
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + context_snippets},
            {"role": "user", "content": f"Error: {user_query}"}
        ],
        options={"temperature": 0}
    )

    answer = response.get("message", {}).get("content", "No response received.")
    print("\n--- LLM RESPONSE ---\n")
    print(answer)


########################################################################
# 5. MAIN EXECUTION
########################################################################

def main() -> None:
    """
    Main pipeline execution.
    """
    # Generate embeddings for data files
    generate_embeddings(DATA_FOLDER)
    # Retrieve context and answer user query
    retrieve_and_answer_query(DATA_FOLDER)


if __name__ == "__main__":
    main()

