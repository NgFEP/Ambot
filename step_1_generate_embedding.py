import os
import json
import numpy as np
from numpy.linalg import norm
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
import ollama
import re

# Normalize a vector to unit length
def normalize_vector(vector):
    norm_value = np.linalg.norm(vector)
    if norm_value == 0:
        return vector
    return vector / norm_value

# Save embeddings to a file
def save_embeddings(filename, embeddings):
    base_name = os.path.basename(filename)
    embeddings_dir = "embeddings"
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    serializable_embeddings = [embedding.tolist() for embedding in embeddings]
    with open(f"{embeddings_dir}/{base_name}.json", "w") as f:
        json.dump(serializable_embeddings, f)

# Load embeddings from a file
def load_embeddings(filename):
    base_name = os.path.basename(filename)
    embeddings_path = f"embeddings/{base_name}.json"
    if not os.path.exists(embeddings_path):
        return False
    with open(embeddings_path, "r") as f:
        embeddings_list = json.load(f)
    return [np.array(embedding) for embedding in embeddings_list]

# Extract text from PDF files and chunk it
def parse_pdf_file(filename, chunk_size=500):
    reader = PdfReader(filename)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text = text.strip()
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return [chunk for chunk in chunks if chunk.strip()]

# Extract question-answer pairs from HTML links and chunk them
def parse_html_links(filename, chunk_size=5000):
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
                prompt = f"""
                Extract question-answer pairs from the following text. 
                Each pair should be formatted as:
                Question: [question text]
                Answer: [answer text]

                Text:
                {chunk}
                """
                llm_response = ollama.chat(
                    model="llama3.2:1b",
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0}
                )

                llm_content = llm_response["message"]["content"]
                pattern = r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)"
                matches = re.findall(pattern, llm_content, flags=re.DOTALL | re.IGNORECASE)

                for (q_text, a_text) in matches:
                    q_text = q_text.strip()
                    a_text = a_text.strip()
                    pair = f"Question: {q_text}\nAnswer: {a_text}"
                    qa_chunks.append(pair)

        except Exception as e:
            print(f"Error processing link {link}: {e}")

    return qa_chunks

# Generate embeddings for the provided chunks
def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        normalize_vector(ollama.embeddings(model=modelname, prompt=chunk)["embedding"])
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings

# Main Embedding Generation Workflow
def generate_embeddings(data_folder):
    combined_chunks = []
    combined_embeddings = []

    # Track separate counts
    pdf_chunk_count = 0
    html_chunk_count = 0

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_name.startswith("."):
            continue

        if file_name == "qa.txt":
            chunks = parse_qa_file(file_path)
        elif file_name == "Amber24.pdf":
            chunks = parse_pdf_file(file_path, chunk_size=500)
            pdf_chunk_count += len(chunks)
        elif file_name == "amber_archive_urls.txt":
            chunks = parse_html_links(file_path, chunk_size=5000)
            html_chunk_count += len(chunks)
        else:
            print(f"Skipping unsupported file format: {file_name}")
            continue

        embeddings = get_embeddings(file_path, "nomic-embed-text", chunks)
        combined_chunks.extend(chunks)
        combined_embeddings.extend(embeddings)

    # Print separate and total chunk counts
    print(f"PDF chunks extracted: {pdf_chunk_count}")
    print(f"HTML chunks extracted: {html_chunk_count}")
    print(f"Total PDF+HTML chunks extracted: {pdf_chunk_count + html_chunk_count}")
    return combined_chunks, combined_embeddings

if __name__ == "__main__":
    data_folder = "data"
    generate_embeddings(data_folder)

