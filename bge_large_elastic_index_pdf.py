import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import glob
import time

# Load Sentence-Transformers model (Locally hosted BAAI/bge-large-en-v1.5)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Ensure the model is locally available

# Custom wrapper class for Sentence-Transformers to integrate with LangChain
class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, documents):
        """Embed a list of documents using the Sentence-Transformers model."""
        document_texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(document_texts)
        return np.array(embeddings)

    def embed_query(self, query):
        """Embed a single query using the Sentence-Transformers model."""
        return np.array(self.model.encode([query]))[0]  # Return the first (and only) element in the array

def load_pdf(pdf_path):
    """Load PDF document using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def split_text(documents):
    """Split text into chunks and wrap them in Document objects."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    split_texts = text_splitter.split_documents(documents)
    wrapped_docs = [Document(page_content=str(chunk)) for chunk in split_texts]
    return wrapped_docs

def create_index_if_needed(es_client, index_name, vector_dim):
    """Ensure that the Elasticsearch index exists and is configured for vector search."""
    if not es_client.indices.exists(index=index_name):
        print(f"Index '{index_name}' does not exist. Creating it...")
        es_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "dense_vector",
                            "dims": vector_dim
                        },
                        "content": {
                            "type": "text"
                        },
                        "source": {
                            "type": "keyword"
                        }
                    }
                }
            }
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Index '{index_name}' already exists.")

def store_embeddings_in_elasticsearch(es_client, index_name, documents, batch_size):
    """Store embeddings and document data in Elasticsearch."""
    embeddings = SentenceTransformerEmbeddings(model)
    doc_texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(documents)

    # Prepare documents for Elasticsearch bulk insertion
    actions = []
    for i, (embedding, content) in enumerate(zip(doc_embeddings, doc_texts)):
        action = {
            "_index": index_name,
            "_id": i,  # Use sequential integers as document IDs
            "_source": {
                "vector": embedding.tolist(),
                "content": content,
                "source": f"doc_{i}"
            }
        }
        actions.append(action)

    # Insert documents in batches
    for i in range(0, len(actions), batch_size):
        batch = actions[i:i + batch_size]
        try:
            bulk(es_client, batch)
            print(f"Inserted batch {i // batch_size + 1} of {len(actions) // batch_size + 1} into index '{index_name}'")
        except Exception as e:
            print(f"Error inserting batch: {e}")
        time.sleep(1)  # Optional: Sleep to avoid overloading Elasticsearch

def process_pdf_in_folder(folder_path, es_client, index_name="pdf_documents", batch_size=10):
    """Process all PDF files in a folder and store embeddings in Elasticsearch."""
    vector_dim = 1024  # BAAI/bge-large-en-v1.5 model output size

    # Ensure the index exists
    create_index_if_needed(es_client, index_name, vector_dim)

    # Fetch all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print("No PDF files found in the folder.")
        return

    # Process each PDF file and add it to Elasticsearch
    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file}")
        documents = load_pdf(pdf_file)
        print(f"Loaded {len(documents)} pages from the PDF.")

        split_docs = split_text(documents)
        print(f"Split into {len(split_docs)} text chunks.")

        store_embeddings_in_elasticsearch(es_client, index_name, split_docs, batch_size)

    print(f"All PDFs in '{folder_path}' have been processed and stored in Elasticsearch.")

if __name__ == "__main__":
    folder_path = "./langchain/input_doc"  # Path to folder containing PDF files
    es_client = Elasticsearch(hosts=["http://localhost:9200"], http_auth=("user", "password"))  # Replace with your Elasticsearch host and credentials
    process_pdf_in_folder(folder_path, es_client, index_name="pdf_documents", batch_size=10)
