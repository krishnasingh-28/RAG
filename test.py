import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads .env if present

# --- Required environment variables ---
# OPENAI_API_KEY=...
# PINECONE_API_KEY=...
# PINECONE_INDEX_NAME=langchain-vector-1024

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# --- FIX: Using 'rag' as the index name to match your screenshot ---
# You can change this back if 'langchain-vector-1024' is correct
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag")

assert OPENAI_API_KEY, "Missing OPENAI_API_KEY"
assert PINECONE_API_KEY, "Missing PINECONE_API_KEY"

# 1) Load documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- IMPORTANT: Ensure this 'documents' folder exists and contains your PDF ---
DOCS_DIR = "documents"  # folder with PDFs
loader = PyPDFDirectoryLoader(DOCS_DIR)
docs = loader.load()
print(f"Loaded pages: {len(docs)}")

# 2) Chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
splits = text_splitter.split_documents(docs)
print(f"Chunks: {len(splits)}")

# 3) Embeddings: OpenAI configured for 1024-dim output
from langchain_openai import OpenAIEmbeddings

# ---
# *** MODIFICATION: Swapped Gemini for OpenAI Embeddings ***
# We are using "text-embedding-3-small", a modern OpenAI model
# that can be configured for 1024 dimensions using the `dimensions`
# parameter. This matches your existing 1024D Pinecone index.
# ---
embeddings_model = "text-embedding-3-small"
EMBED_DIM = 1024  # must match embedding dimension

embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    dimensions=EMBED_DIM  # Request 1024 dimensions
    # openai_api_key is read from env var OPENAI_API_KEY by default
)

# Quick test
vec = embeddings.embed_query("hello world")
print(f"Test embed dim: {len(vec)} (expect 1024)")

# 4) Pinecone setup + index (dimension=1024)
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=PINECONE_API_KEY)

# EMBED_DIM is already defined in section 3

existing = [x["name"] for x in pc.list_indexes()]
if INDEX_NAME not in existing:
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Creating Pinecone index '{INDEX_NAME}' with dim={EMBED_DIM}...")
else:
    print(f"Using existing Pinecone index '{INDEX_NAME}'")

index = pc.Index(INDEX_NAME)

# 5) LangChain vector store wrapper
# ---
# *** FIX 2: Corrected the ImportError ***
# The import `from langchain_pinecone import PineconeVectorStore` was failing.
# This uses the stable `Pinecone` class from `langchain_community` and
# aliases it as `PineconeVectorStore` so the rest of your code
# (vectorstore = PineconeVectorStore.from_documents(...)) works perfectly.
# ---
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    index_name=INDEX_NAME,
)
print(f"Upserted {len(splits)} chunks to Pinecone (1024D).")

# 6) Retrieval helper
def similarity_search(query: str, k: int = 2):
    return vectorstore.similarity_search(query, k=k)

# 7) Retrieval-only baseline answer
def retrieve_only_answer(query: str, k: int = 2):
    docs = similarity_search(query, k)
    return "\n\n".join([d.page_content for d in docs])

# 8) Example query (from the Budget PDF)
print("\n=== Retrieval-only demo (1024D) ===")
# Query from the PDF you provided
q1 = "What is the GDP growth rate for FY 2024?"
print(f"Query: {q1}")
print("Answer:")
print(retrieve_only_answer(q1))

q2 = "What is the projected size of the semiconductor market by 2026?"
print(f"\nQuery: {q2}")
print("Answer:")
print(retrieve_only_answer(q2))