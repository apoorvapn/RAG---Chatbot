"""
ingest.py
---------
Stage 1 of the RAG pipeline: build the knowledge base.

What this does:
1. Loads every .pdf and .txt file from ./data
2. Splits each document into overlapping chunks (so retrieval stays precise
   even for long documents, since LLM context windows are limited)
3. Converts each chunk into a vector embedding
4. Stores the vectors in a local FAISS index on disk (./faiss_index)

Run this once whenever you add/change files in ./data:
    python ingest.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from a local .env file, if present

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = Path(__file__).parent / "faiss_index"

CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 150    # overlap between consecutive chunks, to preserve context across boundaries


def load_documents():
    """Load every supported file in ./data into LangChain Document objects."""
    docs = []
    for path in DATA_DIR.glob("*"):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        elif path.suffix.lower() == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        else:
            continue
        loaded = loader.load()
        print(f"  loaded {len(loaded)} page(s)/doc(s) from {path.name}")
        docs.extend(loaded)
    return docs


def chunk_documents(docs):
    """Split documents into smaller overlapping chunks for accurate retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_index(chunks):
    """Embed the chunks and persist a FAISS vector index to disk."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_DIR))
    return vectorstore


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: Set the OPENAI_API_KEY environment variable first.\n"
            "  export OPENAI_API_KEY=sk-...   (Mac/Linux)\n"
            "  setx OPENAI_API_KEY sk-...     (Windows)"
        )

    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents()
    if not docs:
        raise SystemExit(f"No .pdf or .txt files found in {DATA_DIR}. Add some and re-run.")
    print(f"Loaded {len(docs)} raw document(s)/page(s).")

    print("Splitting into chunks ...")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

    print("Embedding chunks and building FAISS index ...")
    build_index(chunks)
    print(f"Done. Index saved to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
