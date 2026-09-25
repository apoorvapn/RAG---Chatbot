"""
chat.py
-------
Stage 2 of the RAG pipeline: query the knowledge base.

For each question:
1. Embed the question with the same embedding model used during ingestion
2. Do a similarity search against the FAISS index to retrieve the top-k
   most relevant chunks (this is the "Retrieval" in RAG)
3. Insert those chunks into a prompt as context
4. Send the prompt to the LLM, which generates an answer grounded in the
   retrieved context (this is the "Augmented Generation")
5. Print the answer along with which source chunks it was based on

Run:
    python chat.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from a local .env file, if present

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

INDEX_DIR = Path(__file__).parent / "faiss_index"
TOP_K = 4  # how many chunks to retrieve per question

PROMPT_TEMPLATE = """You are a helpful assistant answering questions using ONLY the
context below. If the answer isn't in the context, say you don't know instead of
guessing.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    """Join retrieved chunks into a single context string, and remember them for citation."""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "ERROR: Set the OPENAI_API_KEY environment variable first.\n"
            "  export OPENAI_API_KEY=sk-..."
        )
    if not INDEX_DIR.exists():
        raise SystemExit("No FAISS index found. Run `python ingest.py` first.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chatbot ready. Type a question, or 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        # Retrieve first (separately) so we can show sources alongside the answer
        retrieved_docs = retriever.invoke(question)

        answer = rag_chain.invoke(question)
        print(f"\nBot: {answer}\n")

        print("Sources used:")
        for i, d in enumerate(retrieved_docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            loc = f"{src} (page {page})" if page is not None else src
            preview = d.page_content[:100].replace("\n", " ")
            print(f"  [{i}] {loc} — \"{preview}...\"")
        print()


if __name__ == "__main__":
    main()
