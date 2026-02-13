# ingest.py

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# Load document
#
loader = TextLoader("data/policy.txt")
documents = loader.load()

print("✅ Document loaded")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"✅ Created {len(chunks)} chunks")

# 🔥 Use GEMMA 2B for embeddings
embeddings = OllamaEmbeddings(model="gemma:2b")

# Store in Chroma
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("✅ Stored in chroma_db")
