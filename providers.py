"""
providers.py
------------
Picks which embedding model + LLM to use, so the rest of the code doesn't care.

  PROVIDER=ollama  (default) -> free, open-source models running on your laptop
  PROVIDER=openai            -> paid OpenAI API (needs OPENAI_API_KEY)

Set it in your .env file. Each provider gets its own FAISS index folder,
because different embedding models produce incompatible vectors.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("PROVIDER", "ollama").strip().lower()

OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")

INDEX_DIR = Path(__file__).parent / f"faiss_index_{PROVIDER}"


def check_config():
    """Fail early with a friendly message if the setup is incomplete."""
    if PROVIDER not in {"ollama", "openai"}:
        raise SystemExit(f"PROVIDER must be 'ollama' or 'openai', got '{PROVIDER}'.")
    if PROVIDER == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("PROVIDER=openai needs OPENAI_API_KEY in your .env file.")


def get_embeddings():
    if PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)


def get_llm():
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_CHAT_MODEL, temperature=0)
