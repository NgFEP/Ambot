import os
import json
import logging
import re
import time
from typing import List, Tuple, Optional
import asyncio
import aiohttp
import numpy as np
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import faiss
import ollama

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
PDF_CHUNK_SIZE = 100  
HTML_CHUNK_SIZE = 100  
EMBEDDINGS_DIR = "embeddings"  
DATA_FOLDER = "data"  

# LLM Models
LLM_MODEL = "deepseek-r1:14b"  
EMBEDDING_MODEL = "nomic-embed-text"  

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm_value = np.linalg.norm(vector)
    return vector if norm_value == 0 else vector / norm_value

def save_embeddings(filename: str, embeddings: List[np.ndarray], chunks: List[str]) -> None:
    base_name = os.path.basename(filename)
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    with open(os.path.join(EMBEDDINGS_DIR, f"{base_name}.json"), "w") as f:
        json.dump([emb.tolist() for emb in embeddings], f)

    with open(os.path.join(EMBEDDINGS_DIR, f"{base_name}_chunks.json"), "w") as f:
        json.dump(chunks, f)

    logging.info(f"Saved embeddings and chunks for {base_name}")

def load_embeddings(filename: str) -> Optional[List[np.ndarray]]:
    base_name = os.path.basename(filename)
    path = os.path.join(EMBEDDINGS_DIR, f"{base_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [np.array(emb) for emb in json.load(f)]

def parse_pdf_file(filename: str, start_page: int = 15, end_page: int = 15) -> List[str]:
    reader = PdfReader(filename)
    text = " ".join(
        reader.pages[i].extract_text() 
        for i in range(start_page - 1, end_page) if reader.pages[i].extract_text()
    )
    
    words = text.split()
    return [" ".join(words[i:i + PDF_CHUNK_SIZE]) for i in range(0, len(words), PDF_CHUNK_SIZE)]

async def fetch_html_content(url: str) -> Optional[str]:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                return await response.text() if response.status == 200 else None
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return None

async def parse_html_links(filename: str) -> List[str]:
    with open(filename, encoding="utf-8-sig") as f:
        links = f.read().strip().splitlines()

    tasks = [fetch_html_content(link) for link in links]
    html_contents = await asyncio.gather(*tasks)
    return ["\n".join(BeautifulSoup(content, "html.parser").stripped_strings) for content in html_contents if content]

def get_embeddings(filename: str, modelname: str, chunks: List[str]) -> List[np.ndarray]:
    embeddings = load_embeddings(filename)
    if embeddings is not None:
        return embeddings

    embeddings = [
        normalize_vector(ollama.embeddings(model=modelname, prompt=chunk)["embedding"])
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings, chunks)
    return embeddings

async def generate_embeddings(data_folder: str) -> None:
    all_chunks, all_embeddings = [], []

    for file_name in os.listdir(data_folder):
        if file_name.startswith("."):
            continue

        file_path = os.path.join(data_folder, file_name)
        chunks = []

        if file_name.endswith(".pdf"):
            chunks = parse_pdf_file(file_path)
        elif file_name == "amber_archive_urls.txt":
            chunks = await parse_html_links(file_path)
        
        if chunks:
            embeddings = get_embeddings(file_path, EMBEDDING_MODEL, chunks)
            all_chunks.extend(chunks)
            all_embeddings.extend(embeddings)

    if all_embeddings:
        index = faiss.IndexFlatL2(all_embeddings[0].shape[0])
        index.add(np.array(all_embeddings).astype('float32'))
        faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "faiss_index.bin"))
        logging.info("FAISS index built and saved.")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(generate_embeddings(DATA_FOLDER))
    logging.info(f"Execution time: {time.time() - start_time:.2f} seconds")

