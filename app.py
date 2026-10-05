"""
app.py
------
FastAPI web server that wraps the RAG pipeline and serves the chat UI.

Run:
    uvicorn app:app --reload
Then open http://localhost:8000 in your browser.
"""

from pathlib import Path
import asyncio
import json
import threading

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from providers import INDEX_DIR, PROVIDER, check_config, get_embeddings, get_llm

# ── setup ────────────────────────────────────────────────────────────────────

TOP_K = 4

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the \
context below. If the answer isn't in the context, say you don't know instead of guessing.

When the user asks for a diagram, flowchart, chart, graph, or visual representation, \
produce a valid Mermaid diagram inside a fenced code block tagged as `mermaid`. \
Use the most appropriate Mermaid diagram type (flowchart, sequenceDiagram, classDiagram, \
stateDiagram-v2, erDiagram, etc.). Only use Mermaid when explicitly asked for a visual.

Context:
{context}

Question: {question}

Answer:"""

app = FastAPI(title="RAG Chatbot API")

# Serve everything under ./static as static files
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── lazy-load the RAG chain once on first request ────────────────────────────

_rag_chain = None
_retriever = None


def get_rag():
    global _rag_chain, _retriever
    if _rag_chain is None:
        check_config()
        if not INDEX_DIR.exists():
            raise RuntimeError(
                "No FAISS index found. Run `python ingest.py` first."
            )
        embeddings = get_embeddings()
        vectorstore = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        _retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        def format_docs(docs):
            return "\n\n---\n\n".join(d.page_content for d in docs)

        _rag_chain = (
            {"context": _retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
    return _rag_chain, _retriever


# ── routes ───────────────────────────────────────────────────────────────────


@app.get("/", response_class=FileResponse)
async def serve_ui():
    """Serve the chat UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(str(index_file))


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    location: str
    preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    provider: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Run the RAG pipeline and return the answer + source citations."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        rag_chain, retriever = get_rag()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    retrieved_docs = retriever.invoke(question)
    answer = rag_chain.invoke(question)

    sources: list[Source] = []
    for d in retrieved_docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page")
        location = f"{src} (page {page})" if page is not None else src
        preview = d.page_content[:150].replace("\n", " ")
        sources.append(Source(location=location, preview=preview))

    return ChatResponse(answer=answer, sources=sources, provider=PROVIDER)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """Stream RAG response token-by-token via Server-Sent Events."""
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        rag_chain, retriever = get_rag()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # ── Run retrieval + streaming in a background thread ──────
        def run_stream():
            try:
                docs = retriever.invoke(question)
                # Put retrieved docs info into the queue first
                sources = []
                for d in docs:
                    src = d.metadata.get("source", "unknown")
                    page = d.metadata.get("page")
                    location = f"{src} (page {page})" if page is not None else src
                    preview = d.page_content[:150].replace("\n", " ")
                    sources.append({"location": location, "preview": preview})

                for chunk in rag_chain.stream(question):
                    asyncio.run_coroutine_threadsafe(
                        queue.put(("token", chunk)), loop
                    )
                asyncio.run_coroutine_threadsafe(
                    queue.put(("sources", sources)), loop
                )
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("error", str(exc))), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(
                    queue.put(("done", None)), loop
                )

        t = threading.Thread(target=run_stream, daemon=True)
        t.start()

        # ── Yield SSE events as they arrive ──────────────────────
        while True:
            kind, value = await queue.get()
            if kind == "token":
                yield f"data: {json.dumps({'type': 'token', 'content': value})}\n\n"
            elif kind == "sources":
                yield f"data: {json.dumps({'type': 'sources', 'sources': value})}\n\n"
            elif kind == "error":
                yield f"data: {json.dumps({'type': 'error', 'content': value})}\n\n"
                break
            elif kind == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "provider": PROVIDER}
