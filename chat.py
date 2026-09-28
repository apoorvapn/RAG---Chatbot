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


from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from providers import INDEX_DIR, PROVIDER, check_config, get_embeddings, get_llm

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
    check_config()
    if not INDEX_DIR.exists():
        raise SystemExit("No FAISS index found. Run `python ingest.py` first.")

    embeddings = get_embeddings()
    vectorstore = FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print(f"RAG chatbot ready (provider: {PROVIDER}). Type a question, or 'exit' to quit.\n")
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
