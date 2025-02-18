# Ambot

**Ambot** is a scalable and robust Retrieval-Augmented Generation (RAG) system designed for the AMBER (https://ambermd.org/) AI chatbot. 
It leverages the **AMBER** manual and Q&A from the AMBER mailing list (http://archive.ambermd.org/) to provide accurate and context-aware responses.

## Purpose
**Ambot** is designed to assist users in resolving technical issues related to **AMBER installation**, **simulation setup**, and **troubleshooting** efficiently. Instead of spending hours searching through online forums, documentation, or the AMBER mailing list, users can directly ask their questions to Ambot. It retrieves precise and relevant information from the AMBER manual and curated Q&A, saving time and providing reliable solutions.

Whether you're facing errors during installation, struggling with simulation setup, or encountering runtime issues, Ambot is here to help. It serves as a one-stop solution for AMBER-related queries, making it easier for researchers and developers to focus on their work rather than troubleshooting.

# Getting Started
## Dependencies
- **Python**: 3.11
- **OS**: Ubuntu 22.04.4 LTS
- **pip**: 23.2.1
- **Nvidia Driver Version**: 555.42
- **CUDA Version**: 12.5, 12.6, 12.8

## LLM Providers
- deepseek-r1:14b
## Text embed provider
- nomic-embed-text:latest

# Installation Instructions (Ubuntu)
Follow these steps to install and set up Ambot:
- Open your terminal and run the command to clone the repository.
```
git clone https://github.com/NgFEP/Ambot.git
```
- Change the directory
```
cd Ambot
```
```
python3 -m venv Ambot_env
```
```
source Ambot_env/bin/activate
```
- Ensure you have Python 3.11+ installed, then run:
```
pip install -r requirements.txt
```

- Download Ollama (https://ollama.com/download)
```
curl -fsSL https://ollama.com/install.sh | sh
```
These commands are used to manage the Ollama service (likely a machine learning model server) on a Linux system using `systemctl`, 
the systemd service manager.(https://github.com/ollama/ollama/blob/main/docs/linux.md)

```
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
sudo systemctl status ollama
```
To pull the `deepseek-r1:14b` and `nomic-embed-text:latest` models using Ollama,

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
- To generate embeddings and create Ambot:
```
python embedding_generation.py
```
## Step 2

- To query and test **Ambot**:
```
python query_retrieval.py
```
You'll be prompted to enter a query:

Enter your query (or type 'exit' to quit):

For an example:
![alt text](https://github.com/NgFEP/Ambot/blob/main/Screenshot%20from%202025-02-17%2014-17-40.png)

# Changing the Large Language Model (LLM)
To switch from the "deepseek-r1:14b" LLM to "Llama 3.2", you need to modify the LLM_MODEL setting in embedding_generation.py and query_retrieval.py. 
```
LLM_MODEL = "deepseek-r1:14b"
```
to 
```
LLM_MODEL = "Llama 3.2"
```

# Installation Instructions (Windows with Anaconda)

Follow these steps to install and set up Ambot on Windows using Anaconda:

1. **Install Anaconda**  
   Download and install Anaconda from [official website](https://docs.anaconda.com/anaconda/install/). if you haven't already.

2. **Install CUDA Toolkit and Add to System PATH**  
   - Download **CUDA Toolkit** from [NVIDIA](https://developer.nvidia.com/cuda-downloads).
   - Choose the version 12.8, 12.6 or 12.5 preferably.
   - During installation, select **Custom Options** and ensure **Visual Studio Integration** is enabled.
   - After installation, add CUDA binaries to the system's PATH:
     1. Press `Win + S`, type **Environment Variables**, and open it.
     2. In **System Properties**, click **Environment Variables**.
     3. Under **User Variables for ...**, find and edit the **Path** variable.
     4. Click **New**, then add the following paths (adjust for your CUDA version):
     	If the version installed is v12.8 (adjust according to the version installed on your machine)

        ```bash
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
        ```

     5. Click **OK** to save changes.

   - **Verify CUDA Installation**:
     Windows key + R (to open the Run dialog), then type "cmd" and press Enter to open command prompt, type
     ```bash
     nvcc --version
     ```
     ```bash
     nvidia-smi
     ```
   - **Restart your computer**:
     Restart your system to apply the updates.

3. **Open Anaconda Prompt**  
   Search for "Anaconda Prompt" in the Start menu and open it.

4. **Create a New Conda Environment**  
    Create a new environment with Python 3.11:
    ```bash
    conda create -n Ambot_env python=3.11
    ```
5. **Activate the Environment**  
    Activate the newly created environment:
    ```bash
    conda activate Ambot_env
    ```
6. **Clone the Repository**  
    Clone the Ambot repository:
    ```bash
    git clone https://github.com/NgFEP/Ambot.git
    ```
7. **Change Directory**  
    Navigate to the cloned repository:
    ```bash
    cd Ambot
    ```
8. **Install Dependencies**  
    Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
9. **Install Ollama**  
    install Ollama:
    ```bash
    conda install conda-forge::ollama
    ```
10. **Pull Models**  
    Pull the required models using Ollama:
    ```bash
    ollama pull deepseek-r1:14b
    ollama pull nomic-embed-text:latest
    ```
11. **Create Embeddings Folder**  
    Create a folder in case it's not there already to store the embeddings:
    ```bash
    mkdir embeddings
    ```
12. **Run the Scripts**  
    To generate embeddings and create **Ambot**:
    ```bash
    python embedding_generation.py
    ```
    To query and test **Ambot**:
    ```bash
    python query_retrieval.py
    ```
    Ask your technical questions. Example:
    How do I analyze the results of an AMBER simulation using cpptraj?

Rest similar to **Installation Instructions (Ubuntu)**

💖 If you like this project, give it a star! ⭐

