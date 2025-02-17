# AMBOT

AMBOT is a scalable and robust Retrieval-Augmented Generation (RAG) system designed for the AMBER (https://ambermd.org/) AI chatbot. 
It leverages the AMBER manual and Q&A from the AMBER mailing list (http://archive.ambermd.org/) to provide accurate and context-aware responses.

# Getting Started
## Dependencies
- **Python**: 3.11
- **OS**: Ubuntu 22.04.4 LTS
- **pip**: 23.2.1
- **Nvidia Driver Version**: 555.42
- **CUDA Version**: 12.5

## LLM Providers



# Installation Instructions
Follow these steps to install and set up AMBOT:
- Open your terminal and run the command to clone the repository.
```
git clone https://github.com/NgFEP/Ambot.git
```
- Change the directory
```
cd Ambot
```
```
python3 -m venv AMBOT_env
```
```
source AMBOT_env/bin/activate
```
- Ensure you have Python 3.11+ installed, then run:
```
pip install -r requirements.txt
```

- Download Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```
These commands are used to manage the Ollama service (likely a machine learning model server) on a Linux system using `systemctl`, the systemd service manager.

```
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```
```
ollama pull deepseek-r1:14b
```
```
ollama pull nomic-embed-text:latest
```
- Create a folder to store the embeddings.
```
mkdir -p embeddings
```
# Usage
## Step 1
- Run embedding_generation.py to process PDFs and web content:
```
python embedding_generation.py
```
## Step 2

- Run query_retrieval.py to query the stored knowledge base:
```
python query_retrieval.py
```
You'll be prompted to enter a query:

Enter your query (or type 'exit' to quit):

For an example:
![alt text](https://github.com/NgFEP/Ambot/blob/main/Screenshot%20from%202025-02-17%2014-17-40.png)
Enter your query (or type 'exit' to quit): what is AMBER?



💖 If you like this project, give it a star! ⭐

