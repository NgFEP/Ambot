import os, ollama
import json
import numpy as np
from numpy.linalg import norm
from flask import Flask, request, jsonify, render_template
from step_1_generate_embedding import parse_pdf_file
from step_1_generate_embedding import parse_html_links


# Initialize Flask app
app = Flask(__name__)

# Global variables for chunks and embeddings
combined_chunks = []
combined_embeddings = []

# Function to load embeddings
def load_embeddings(filename):
    base_name = os.path.basename(filename)
    embeddings_path = f"embeddings/{base_name}.json"
    if not os.path.exists(embeddings_path):
        return False
    with open(embeddings_path, "r") as f:
        embeddings_list = json.load(f)
    return [np.array(embedding) for embedding in embeddings_list]

# Function to normalize vectors
def normalize_vector(vector):
    norm_value = np.linalg.norm(vector)
    if norm_value == 0:
        return vector
    return vector / norm_value

# Function to find most similar embeddings
def find_most_similar(query_embedding, embedding_list, top_k=5):
    query_norm = norm(query_embedding)
    similarities = []
    for idx, emb in enumerate(embedding_list):
        score = np.dot(query_embedding, emb) / (query_norm * norm(emb))
        similarities.append((score, idx))
    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities[:top_k]

# Load data into memory
def load_data():
    global combined_chunks, combined_embeddings
    data_folder = "/media/saikat/lenovo_4tb/saikat_work/RAG_project/test_rag_llm_no_vcdb_host/data/"
    combined_chunks = []
    combined_embeddings = []

    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_name.startswith("."):
            continue

        if file_name == "qa.txt":
            chunks = parse_qa_file(file_path)
        elif file_name.endswith(".pdf"):
            chunks = parse_pdf_file(file_path, chunk_size=500)
        elif file_name == "amber_archive_urls.txt":
            chunks = parse_html_links(file_path, chunk_size=5000)
        else:
            print(f"Skipping unsupported file format: {file_name}")
            continue

        embeddings = load_embeddings(file_path)
        if not embeddings:
            print(f"No embeddings found for {file_name}. Skipping.")
            continue

        combined_chunks.extend(chunks)
        combined_embeddings.extend(embeddings)

    print(f"Total chunks loaded: {len(combined_chunks)}")
    print(f"Total embeddings loaded: {len(combined_embeddings)}")

# Load data when the application starts
load_data()

########################################################################
# Flask Routes
########################################################################

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    global combined_chunks, combined_embeddings

    # Check if embeddings are loaded
    if not combined_embeddings or not combined_chunks:
        return jsonify({"error": "Embeddings or chunks are not loaded."})

    # Get question from request
    question = request.form.get('question')
    if not question:
        return jsonify({"error": "No question provided."})

    # Generate embedding for the question
    question_embedding = normalize_vector(ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"])

    # Find most similar chunks
    similar_chunks = find_most_similar(question_embedding, combined_embeddings, top_k=3)

    # Retrieve relevant chunks
    relevant_chunks = []
    for score, idx in similar_chunks:
        if idx < len(combined_chunks):
            relevant_chunks.append(combined_chunks[idx])

    if not relevant_chunks:
        return jsonify({"error": "No relevant context found."})

    # Combine relevant chunks
    context = "\n".join(relevant_chunks)

    # Generate response using the context
    response = ollama.chat(
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant.\n" + context},
            {"role": "user", "content": question},
        ],
        options={"temperature": 0}  # Example: controlling temperature
    )

    return jsonify({"question": question, "answer": response["message"]["content"]})

if __name__ == '__main__':
    app.run(debug=True)

