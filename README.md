#AMBOT

































🧠 AMBOT (RAG-Based Retrieval System )

📢 A powerful, scalable Retrieval-Augmented Generation (RAG) system using FAISS and DeepSeek LLM.

🛠 Ideal for knowledge-based Q&A, document retrieval, and AI chatbots.

💡 Supports PDFs and web page processing with fast FAISS-based retrieval.

📌 Table of Contents

📌 Features

🚀 Installation

👤 Project Structure

🛠️ Usage

🔧 Configuration

⚡ Performance Optimizations

📌 Features

✅ Retrieval-Augmented Generation (RAG) – Uses FAISS for context-aware query answering.

✅ Multi-Document Support – Processes PDFs and web pages.

✅ DeepSeek-Style Prompting – Ensures structured, fact-based responses.

✅ Fast & Scalable Retrieval – FAISS enables rapid similarity search.

✅ Cosine Similarity Ranking – Retrieves the most relevant chunks.

✅ Cleans LLM Output – Removes unwanted artifacts like <think>.

🚀 Installation

1️⃣ Clone the Repository

git clone https://github.com/NgFEP/Ambot.git

cd Ambot

2️⃣ Install Dependencies

Ensure you have Python 3.8+ installed, then run:

pip install -r requirements.txt

If you don't have a requirements.txt, manually install:

pip install numpy faiss-cpu aiohttp PyPDF2 beautifulsoup4 ollama

3️⃣ Set Up Directory Structure

Create the necessary folders:

mkdir -p embeddings data

👤 Project Structure

📦 RAG_Project

├── 📂 embeddings/               # Stores generated embeddings

├── 📂 data/                     # Stores input PDF & web page files

├── 📜 embedding_generation.py   # Generates embeddings from PDFs & HTML

├── 📜 query_retrieval.py        # Retrieves and answers queries

├── 📜 README.md                 # Documentation

🛠️ Usage

👉 Step 1: Generate Embeddings

Run embedding_generation.py to process PDFs and web content:

python embedding_generation.py

This script:

Extracts text from PDFs and HTML.

Splits text into chunks.

Generates embeddings for each chunk.

Stores embeddings in FAISS for fast retrieval.

👉 Step 2: Retrieve Answers

Run query_retrieval.py to query the stored knowledge base:

python query_retrieval.py

You'll be prompted to enter a query:

Enter your query (or type 'exit' to quit): What is the impact of inflation?

The system will:

1️⃣ Compute the query embedding.

2️⃣ Search FAISS for the most relevant chunks.

3️⃣ Use DeepSeek LLM to generate an answer.

4️⃣ Return a clean, structured response.

🔧 Configuration

Modify model settings in both scripts:

# Constants
EMBEDDINGS_DIR = "embeddings"  
DATA_FOLDER = "data"  
LLM_MODEL = "deepseek-r1:14b"  
EMBEDDING_MODEL = "nomic-embed-text"  

To change how many retrieved chunks are used for answering:

top_k = 3  # Change this to retrieve more or fewer chunks

⚡ Performance Optimizations

✅ Uses FAISS for fast similarity search.✅ Async web scraping speeds up data extraction.✅ Text chunking ensures better LLM processing.✅ Cosine similarity ranking improves accuracy.


💖 If you like this project, give it a star! ⭐

