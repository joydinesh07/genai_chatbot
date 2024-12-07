# README: PDF Processing and Querying with Qdrant and Sentence-Transformers

This repository contains two Python scripts for processing PDFs, storing them in a vector database (Qdrant), and building a chatbot interface for querying the stored documents using natural language. 

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Usage](#usage)
5. [How It Works](#how-it-works)
6. [Scripts Overview](#scripts-overview)
7. [Configuration](#configuration)
8. [Notes](#notes)

---

## Features

- **PDF Processing**:
  - Extract text from PDF files.
  - Split the extracted text into smaller, manageable chunks.
  
- **Vectorization**:
  - Convert text into embeddings using `SentenceTransformer` (`all-MiniLM-L6-v2`).

- **Vector Store Management**:
  - Create collections in Qdrant.
  - Store and retrieve vectors in/from Qdrant.

- **Chatbot Interface**:
  - Query stored documents in natural language.
  - Retrieve the most relevant document snippets with similarity scores.

---

## Requirements

- Python 3.8+
- Libraries:
  - `langchain_community`
  - `langchain`
  - `sentence_transformers`
  - `qdrant_client`
  - `numpy`
  - `PyPDFLoader`
  - `glob`

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/joydinesh07/genai_chatbot.git
   cd genai_chatbot
   ```

2. Install required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure Qdrant is accessible. Set up your Qdrant API key and URL in the script configuration.

---

## Usage

### **Step 1: Process PDFs**

Run the `index_pdf.py` script to process and store PDF files in the Qdrant vector store:

```bash
python index_pdf.py
```

- Place your PDFs in the folder `./langchain/input_doc`.
- The script will:
  1. Extract text from each PDF.
  2. Split the text into smaller chunks.
  3. Store the chunks as vectors in Qdrant.

### **Step 2: Run the Chatbot**

Run the `chatbot.py` script to start the chatbot interface:

```bash
python chatbot.py
```

- Type your queries to retrieve the most relevant document snippets from the stored data.
- To exit, type `exit`.

---

## How It Works

1. **PDF Processing**:
   - The `PyPDFLoader` extracts text from PDF pages.
   - The text is split into chunks (default: 1000 characters with 10 characters overlap).

2. **Vectorization**:
   - Text chunks are embedded into 384-dimensional vectors using `SentenceTransformer`.

3. **Vector Storage**:
   - Vectors are stored in Qdrant with metadata, including the chunk content and source information.

4. **Querying**:
   - User queries are embedded into vectors.
   - Qdrant retrieves the top `N` (default: 3) closest vectors using cosine similarity.
   - The chatbot combines retrieved snippets into a response.

---

## Scripts Overview

### `index_pdf.py`

- **Purpose**: Process PDFs and upload their embeddings to Qdrant.
- **Key Functions**:
  - `load_pdf()`: Load text from PDFs.
  - `split_text()`: Split extracted text into chunks.
  - `create_vector_store()`: Create or update a collection in Qdrant with vectorized text.

### `chatbot.py`

- **Purpose**: Query the Qdrant vector store and generate responses.
- **Key Functions**:
  - `query_qdrant()`: Search the Qdrant collection for relevant vectors.
  - `generate_response()`: Generate chatbot responses based on retrieved vectors.

---

## Configuration

### Qdrant Connection
- URL and API key are defined in the scripts:
  ```python
  client = QdrantClient(
      url="https://your-qdrant-instance-url",
      api_key="your-api-key"
  )
  ```

### Folder Paths
- Modify the `folder_path` variable in `index_pdf.py` to point to your folder containing PDFs:
  ```python
  folder_path = "./langchain/input_doc"
  ```

---

## Notes

- Ensure the Sentence-Transformers model is downloaded before running the scripts.
- If your Qdrant collection size changes frequently, consider reindexing the PDF folder periodically.
- Batch size for inserting documents into Qdrant can be adjusted for performance.

---

Enjoy building your AI-powered PDF document search and query system! 🎉