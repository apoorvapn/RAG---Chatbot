# RAG Chatbot over Your Own Documents

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using
your own PDFs/notes as its knowledge base, built with **LangChain**, a
**FAISS** vector store, and open-source models via **Ollama** (or OpenAI, optionally).

## How it works

```
        ┌────────────────────┐              ┌───────────────────┐
        │  Your PDFs / .txt   │              │   User question     │
        │  files (./data)     │              └─────────┬──────────┘
        └─────────┬──────────┘                          │
                   │ 1. load                             │ 4. embed
                   ▼                                      ▼
        ┌────────────────────┐              ┌───────────────────┐
        │  Split into chunks  │              │  Similarity search  │
        │  (RecursiveChar-    │              │  against FAISS index│
        │   TextSplitter)     │              └─────────┬──────────┘
        └─────────┬──────────┘                          │ 5. top-k chunks
                   │ 2. embed                            ▼
                   ▼                          ┌───────────────────┐
        ┌────────────────────┐               │  Prompt = chunks +  │
        │  FAISS vector index │◄─── 3. save   │  question           │
        │  (faiss_index/)     │               └─────────┬──────────┘
        └────────────────────┘                          │ 6. send to LLM
                                                           ▼
                                              ┌───────────────────┐
                                              │  LLM (llama3.2)     │
                                              │  generates grounded │
                                              │  answer + sources   │
                                              └───────────────────┘
```

**`ingest.py`** (run once, or whenever your source documents change):
1. Loads every `.pdf`/`.txt` file in `./data`
2. Splits each document into overlapping chunks (800 chars, 150 overlap) —
   this keeps chunks small enough to retrieve precisely, while the overlap
   stops important sentences from being cut in half at a chunk boundary
3. Embeds each chunk with `nomic-embed-text` (or OpenAI's `text-embedding-3-small`)
4. Stores all vectors in a local FAISS index (`faiss_index/`)

**`chat.py`** (run every time you want to ask questions):
1. Embeds your question with the same embedding model
2. Runs a similarity search against FAISS to pull the top 4 most relevant
   chunks
3. Inserts those chunks into a prompt template as "context"
4. Sends the prompt to the LLM (`llama3.2` by default), which is instructed to answer only
   from the given context (reduces hallucination)
5. Prints the answer **and** which source chunks it was grounded in

## Setup (100% free, runs locally with Ollama)

1. Install [Ollama](https://ollama.com/download) and make sure it is running.
2. Download the two open-source models (one-time, about 2.5 GB total):

```bash
ollama pull llama3.2           # the chat model (Meta's Llama 3.2, 3B)
ollama pull nomic-embed-text   # the embedding model
```

3. Set up the project:

```bash
git clone <your-repo-url>
cd rag-chatbot
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # defaults to PROVIDER=ollama, no key needed
```

4. Drop your own PDFs/notes into `data/` (a sample file is included), then:

```bash
python ingest.py    # builds the vector index
python chat.py      # ask questions in a loop
```

### Optional: use OpenAI instead
Edit `.env`, set `PROVIDER=openai` and add your `OPENAI_API_KEY`, then re-run
`python ingest.py` (each provider keeps its own index, since embedding models
are not interchangeable).

## Things worth knowing for follow-up questions

- **Why chunk documents at all?** LLMs have a limited context window, and
  embedding a whole long document as one vector blurs together many
  unrelated topics, hurting retrieval accuracy. Smaller chunks retrieve
  more precisely.
- **Why overlap chunks?** Without overlap, a sentence that straddles a
  chunk boundary gets split and loses meaning in both halves.
- **Why FAISS?** It's a fast, local, open-source similarity search library
  (from Meta AI) — good for a project like this where you don't need a
  managed cloud vector DB like Pinecone.
- **What stops the bot from hallucinating?** The prompt explicitly tells
  the LLM to answer only from the retrieved context and to say "I don't
  know" otherwise — plus printing the source chunks makes answers
  auditable.
- **How would you scale this?** Swap FAISS for a managed vector DB
  (Pinecone/Weaviate/Chroma server), add metadata filtering, and consider
  re-ranking retrieved chunks before generation.
- **Extensions you could mention as "next steps":** conversation memory
  (LangChain's `RunnableWithMessageHistory`), a LangGraph agent that
  decides *when* to retrieve vs. answer directly, or wrapping it as an
  MCP server tool.

## Files

```
rag-chatbot/
├── data/                # put your source PDFs/txt files here
│   └── sample_notes.txt # example file so it runs out of the box
├── providers.py         # picks Ollama (free) or OpenAI models
├── ingest.py            # builds the FAISS index
├── chat.py              # query loop (retrieval + generation)
├── requirements.txt
└── README.md
```
