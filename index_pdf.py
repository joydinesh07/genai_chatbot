import os
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import glob
import time

# Load Sentence-Transformers model (Locally hosted)
model = SentenceTransformer('all-MiniLM-L6-v2')

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

def create_vector_store(client, documents, collection_name="pdf_documents", batch_size=25):
    """Create a Qdrant vector store using Sentence-Transformers with batch insertion."""
    embeddings = SentenceTransformerEmbeddings(model)
    
    # Embed documents and prepare them for Qdrant insertion
    doc_texts = [doc.page_content for doc in documents]
    doc_embeddings = embeddings.embed_documents(documents)
    
    # Prepare document IDs as integers
    doc_ids = list(range(len(doc_texts)))  # Generate IDs as integers: 0, 1, 2, ...
    
    # Create list of points with proper structure
    points = []
    for i, (doc_id, embedding, doc_text) in enumerate(zip(doc_ids, doc_embeddings, doc_texts)):
        point = PointStruct(
            id=doc_id,
            vector=embedding.tolist(),
            payload={"source": f"page_{i}", "content": doc_text}
        )
        points.append(point)
    
    # Insert documents in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"Inserted batch {i // batch_size + 1} of {len(points) // batch_size + 1} into collection '{collection_name}'")
        except Exception as e:
            print(f"Error inserting batch: {e}")
        # Optional: Sleep for a brief moment between batches to avoid hitting rate limits
        time.sleep(1)
    
    print(f"Documents uploaded to collection '{collection_name}'")
    return client, collection_name

def process_pdf_in_folder(folder_path, batch_size=10):
    """Process all PDF files in a given folder and store them in Qdrant."""
    # Initialize Qdrant client
    client = QdrantClient(
        url="https://localhost:6333", 
        api_key="ENTER YOUR API KEY",) 
    
    collection_name = "pdf_documents"
    vector_size = 384

    # Function to create collection
    def create_collection_if_needed(collection_name):
        try:
            # Step 1: Get list of existing collections
            existing_collections = client.get_collections()
            existing_collection_names = [col.name for col in existing_collections.collections]
            
            print("Existing collections:", existing_collection_names)

            # Step 2: Check if the collection exists
            if collection_name not in existing_collection_names:
                print(f"Collection '{collection_name}' does not exist. Creating it...")
                
                # Step 3: Create collection if it doesn't exist
                try:
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=vector_size, 
                            distance="Cosine"  # You can also use "Euclidean" or "Dot" based on your use case
                        )
                    )
                    print(f"Collection '{collection_name}' created successfully.")
                    
                    # Give it a second to be fully created in the backend
                    time.sleep(5)
                except Exception as create_error:
                    print(f"Error while creating collection '{collection_name}': {create_error}")
                    return  # Exit if creation failed

            else:
                print(f"Collection '{collection_name}' already exists.")

            # Step 4: Verify collection exists after creation attempt
            existing_collections_after_creation = client.get_collections()
            existing_collection_names_after_creation = [col.name for col in existing_collections_after_creation.collections]
            
            if collection_name in existing_collection_names_after_creation:
                print(f"Collection '{collection_name}' is now available.")
            else:
                print(f"Collection '{collection_name}' still not found after creation.")
        except Exception as e:
            print(f"Error while creating or checking collection: {e}")

            
    # Ensure collection exists before storing embeddings
    create_collection_if_needed(collection_name)


    # Step 2: Fetch all PDF files in the folder
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in the folder.")
        return
    
    # Step 3: Process each PDF file and add it to Qdrant
    for pdf_file in pdf_files:
        print(f"Processing file: {pdf_file}")
        
        # Load and split the PDF
        documents = load_pdf(pdf_file)
        print(f"Loaded {len(documents)} pages from the PDF.")
        
        # Split the document into smaller chunks and wrap them in Document objects
        split_docs = split_text(documents)
        print(f"Split into {len(split_docs)} text chunks.")
        
        # Create vector store using Sentence-Transformers and Qdrant with batch insertion
        client, collection_name = create_vector_store(client, split_docs, collection_name, batch_size)

    print(f"All PDFs in '{folder_path}' have been processed and stored in Qdrant.")

if __name__ == "__main__":
    folder_path = "./langchain/input_doc"  # Replace with the path to your folder containing PDF files
    process_pdf_in_folder(folder_path, batch_size=10)
