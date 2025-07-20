from llama_index.core import VectorStoreIndex, SimpleDirectoryReader,Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openrouter import OpenRouter
import os
from dotenv import load_dotenv

load_dotenv()



# Set the cache directories to your preferred drive (e.g., D:\hf_cache)
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache\transformers"

# Optionally, for datasets (if you use Hugging Face datasets in the future)
os.environ["HF_DATASETS_CACHE"] = r"D:\hf_cache\datasets"


print("--- Configuring LlamaIndex for Hugging Face Embeddings and OpenRouter LLM ---")

print("1. Setting up local embedding model...")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="Qwen/Qwen3-Embedding-0.6B"
)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY environment variable not set.")

print("2. Setting up OpenRouter LLM...")

Settings.llm = OpenRouter(
    api_key=api_key,
    model="moonshotai/kimi-k2:free",
    temperature=0.1
)

print("--- Configuration complete ---")

try:
    # --- Step 3: Load your data ---
    print("Loading documents from the 'data' folder...")
    documents = SimpleDirectoryReader("data").load_data()

    # --- Step 4: Create the index ---
    # This will use the HuggingFaceEmbedding model configured above.
    print("Indexing documents... (This may take a moment)")
    index = VectorStoreIndex.from_documents(documents,chunk_size=512,chunk_overlap=60)
    # --- Step 5: Create a query engine ---
    # This engine will use the OpenRouter LLM configured above.
    print("Setting up query engine...")
    query_engine = index.as_query_engine()
    # --- Step 6: Query your document ---
    print("Querying your document...")
    response = query_engine.query("i want to help human?")
    print("\n--- Response ---")
    print(response)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check that your 'data' folder exists and your OPENROUTER_API_KEY is correct.")

print("Done!")