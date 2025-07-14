from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Define the model name and optimization arguments
model_name = "Qwen/Qwen3-Embedding-0.6B"
model_kwargs = {"device": "cpu"}  # Use 'cpu' if you don't have a GPU
encode_kwargs = {"normalize_embeddings": True} # Recommended for similarity search

# 2. Initialize the HuggingFaceEmbeddings class
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# 3. Use the embedding model within LangChain
# Example: Create a simple vector store
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other.",
    "The Great Wall of China is a series of fortifications.",
]

print("Creating vector store...")
vector_store = FAISS.from_texts(documents, embedding=embeddings)

# 4. Perform a similarity search
query = "What are famous landmarks in China?"
results = vector_store.similarity_search(query)

print("\nQuery:")
print(query)
print("\nTop search results:")
for doc in results:
    print(f"- {doc.page_content}")