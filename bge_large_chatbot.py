import numpy as np  # Import numpy for array operations
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Load Sentence-Transformers model (Locally hosted BAAI/bge-large-en-v1.5)
model = SentenceTransformer('BAAI/bge-large-en-v1.5')  # Ensure the model is locally available

class SentenceTransformerEmbeddings:
    def __init__(self, model):
        self.model = model

    def embed_query(self, query):
        """Embed a single query using the Sentence-Transformers model."""
        return np.array(self.model.encode([query]))[0]  # Return the first (and only) element in the array

def query_qdrant(client, collection_name, query, limit=3):
    """Query the Qdrant vector store to find the most relevant documents."""
    embedding_model = SentenceTransformerEmbeddings(model)
    query_embedding = embedding_model.embed_query(query)  # Embed the query
    
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=limit
    )
    
    results = []
    for result in search_results:
        content = result.payload.get("content", "No content available")
        source = result.payload.get("source", "No source available")
        score = result.score  # This will contain the similarity score (cosine similarity)
        results.append({"content": content, "source": source, "score": score})
    
    return results

def generate_response(documents, query):
    """Generate a response using the most relevant documents."""
    # Combine the most relevant documents into a response, including scores
    context = "\n".join([f"Source: {doc['source']}\nScore: {doc['score']:.4f}\nContent: {doc['content']}" for doc in documents])
    response = f"Based on the query: '{query}', the most relevant content from the documents is:\n\n{context}"
    return response

def chatbot(vector_store_url="https://ecb22197-cfb8-4886-a86c-80218e4ac74f.eu-central-1-0.aws.cloud.qdrant.io:6333", api_key="NnSHRelCDU7Xl-CLRlmq50c0kMc9K0eOP3kShF8TjY_CjrBFyiYsHg"):
    """Chatbot that queries the Qdrant vector store and generates responses."""
    client = QdrantClient(url=vector_store_url, api_key=api_key)
    collection_name = "pdf_documents_bge"
    
    print("Chatbot ready. Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ")
        
        if query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Query the vector store for relevant documents
        search_results = query_qdrant(client, collection_name, query, limit=3)
        
        # Generate response based on the retrieved documents
        response = generate_response(search_results, query)
        
        print("\nChatbot response:")
        print(response)

if __name__ == "__main__":
    chatbot()  # No need for pdf_path here anymore