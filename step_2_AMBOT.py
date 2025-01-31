import os
import json
import numpy as np
from numpy.linalg import norm
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
import ollama
import re

########################################################################
# 1. HELPER FUNCTIONS
########################################################################

def normalize_vector(vector):
    """
    Normalize a vector to unit length.
    """
    norm_value = np.linalg.norm(vector)
    if norm_value == 0:
        return vector
    return vector / norm_value

def load_embeddings(filename):
    """
    Loads pre-generated embeddings from embeddings/<filename>.json.
    Returns a list of NumPy arrays, or False if none found.
    """
    base_name = os.path.basename(filename)
    embeddings_path = os.path.join("embeddings", f"{base_name}.json")
    if not os.path.exists(embeddings_path):
        return False

    with open(embeddings_path, "r") as f:
        embeddings_list = json.load(f)

    return [np.array(embedding) for embedding in embeddings_list]

def find_most_similar(query_embedding, embedding_list, top_k=5):
    """
    Returns the top_k most similar chunk indices (and similarity scores)
    given a query embedding and a list of embeddings.
    """
    query_norm = norm(query_embedding)
    similarities = []
    for idx, emb in enumerate(embedding_list):
        score = np.dot(query_embedding, emb) / (query_norm * norm(emb))
        similarities.append((score, idx))

    # Sort descending by similarity score
    similarities.sort(key=lambda x: x[0], reverse=True)

    # Return the top_k
    return similarities[:top_k]

########################################################################
# 2. CHUNK EXTRACTION (Same as in the 1st script)
#    We only do this again so we know exactly which text matches
#    each embedding index. We do NOT call the embedding API here.
########################################################################

def parse_qa_file(filename):
    """
    Open a .txt file containing Q/A pairs and return list of chunked strings.
    """
    with open(filename, encoding="utf-8-sig") as f:
        content = f.read().strip()
        entries = content.split("Question")[1:]  # Split by 'Question'
        qa_chunks = []
        for entry in entries:
            lines = entry.splitlines()
            question = lines[0].strip()
            try:
                answer_index = lines.index("Answer")
                description = " ".join(lines[2:answer_index]).strip()
                answer = " ".join(lines[answer_index + 1:]).strip()
            except ValueError:
                description = " ".join(lines[2:]).strip()
                answer = "No answer provided."

            chunk = f"Question: {question}\nDescription: {description}\nAnswer: {answer}"
            qa_chunks.append(chunk)
        return qa_chunks

def parse_pdf_file(filename, chunk_size=500):
    """
    Extracts and chunks text from a PDF file.
    """
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text = text.strip()
    words = text.split()
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return [chunk for chunk in chunks if chunk.strip()]

def parse_html_links(filename, chunk_size=5000):
    """
    Extract question-answer pairs from HTML links (as done in the 1st script),
    now chunking the HTML content into fixed sizes.
    """
    with open(filename, encoding="utf-8-sig") as f:
        links = f.read().strip().splitlines()

    qa_chunks = []
    for link in links:
        try:
            response = requests.get(link, timeout=10)
            if response.status_code != 200:
                continue

            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator="\n", strip=True)
            words = page_text.split()
            chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(0, len(words), chunk_size)
            ]

            for chunk in chunks:
                qa_chunks.append(chunk)

        except Exception as e:
            print(f"Error processing link {link}: {e}")

    return qa_chunks

########################################################################
# 3. MAIN SCRIPT: LOAD CHUNKS, LOAD EMBEDDINGS, DO THE QUERY
########################################################################

def main():
    # This system prompt is appended to the final chat prompt
    SYSTEM_PROMPT = """You are a helpful assistant who answers questions 
    based on the provided error. Use the context snippets to construct your answer. 
    Be concise, and if unsure, say you don't know.
    Context:
    """

    data_folder = "data"
    combined_chunks = []
    combined_embeddings = []

    # Loop over files in data_folder again, but this time
    # we only LOAD pre-generated embeddings (no new embedding calls).
    for file_name in os.listdir(data_folder):
        # Skip hidden/system files
        if file_name.startswith("."):
            continue

        file_path = os.path.join(data_folder, file_name)

        # Re-parse the file into text chunks (the same way you did in the 1st script)
        if file_name == "qa.txt":
            chunks = parse_qa_file(file_path)
        elif file_name.endswith(".pdf"):
            chunks = parse_pdf_file(file_path, chunk_size=500)
        elif file_name == "amber_archive_urls.txt":
            chunks = parse_html_links(file_path, chunk_size=5000)
        else:
            print(f"Skipping unsupported file format: {file_name}")
            continue

        # Load the corresponding embeddings
        embeddings = load_embeddings(file_path)
        if embeddings is False:
            print(f"No existing embeddings found for '{file_name}'. "
                  "Please run the 1st script to generate them first.")
            continue

        # Ensure the number of chunks matches the number of embeddings
        if len(chunks) != len(embeddings):
            print(f"Warning: Mismatch in chunks ({len(chunks)}) and embeddings ({len(embeddings)}) for file: {file_name}")
            continue

        # Append them to our combined lists
        combined_chunks.extend(chunks)
        combined_embeddings.extend(embeddings)

    # Debug loaded data
    print(f"Total chunks: {len(combined_chunks)}")
    print(f"Total embeddings: {len(combined_embeddings)}")
    #if combined_chunks:
        #print(f"Sample chunk: {combined_chunks[0][:100]}")
    #if combined_embeddings:
        #print(f"Sample embedding: {combined_embeddings[0][:5]}")

    if not combined_chunks or not combined_embeddings:
        print("No chunks or embeddings found. Make sure you've run the 1st script.")
        return

    # Now, prompt the user for the error (query)
    error = input("Describe the error -> ").strip()
    if not error:
        print("No error provided. Exiting.")
        return

    # Embed the user's query using the same model used before
    prompt_embedding_dict = ollama.embeddings(model="nomic-embed-text", prompt=error)
    query_embedding = normalize_vector(prompt_embedding_dict["embedding"])
    print(f"Query embedding: {query_embedding[:5]}")

    # Find the top-3 most relevant chunks
    top_similar = find_most_similar(query_embedding, combined_embeddings, top_k=3)

    # Debug similarity scores
    print("Similarity Scores:")
    for score, idx in top_similar:
        if idx < len(combined_chunks):
            print(f"Score: {score}, Chunk: {combined_chunks[idx][:100]}")
        else:
            print(f"Warning: Index {idx} out of range for combined_chunks.")

    # Prepare those chunks as context
    relevant_context = []
    for score, idx in top_similar:
        if idx < len(combined_chunks):
            relevant_context.append(combined_chunks[idx])

    if not relevant_context:
        print("No relevant context found.")
        return

    # Combine the top chunks into a single context string
    context_snippets = "\n".join(relevant_context)

    # Build the final conversation for the LLM
    final_system_message = SYSTEM_PROMPT + context_snippets

    # Use Ollama to generate a response based on the context and user error
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": final_system_message},
            {"role": "user", "content": f"Error: {error}"},
        ],
        options={"temperature": 0}  # Example: controlling temperature
    )

    # Print the final answer
    print("\n--- LLM RESPONSE ---\n")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()

