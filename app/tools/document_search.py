import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


load_dotenv()


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"


embeddings = OpenAIEmbeddings(
    model=os.getenv("LM_STUDIO_EMBEDDING_MODEL"),
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    check_embedding_ctx_length=False,
)


def _bm25_preprocess(text):
    """
    Convert text into tokens that BM25 will search.
    Simple lowercase whitespace tokenization.
    """
    return text.lower().split()


def _get_hybrid_retriever(k=5):
    """
    Build and return a hybrid retriever that combines
    vector (semantic) search with BM25 (keyword) search
    using equal weights via EnsembleRetriever.
    """

    # Load the existing Chroma vectorstore
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings,
    )

    # Get all stored documents for BM25 indexing
    stored = vectorstore.get(
        include=["documents", "metadatas"]
    )

    if not stored["documents"]:
        return None

    # Rebuild LangChain Documents for BM25
    from langchain_core.documents import Document

    bm25_docs = [
        Document(
            page_content=doc,
            metadata=meta,
        )
        for doc, meta in zip(
            stored["documents"],
            stored["metadatas"],
        )
    ]

    # Create vector retriever (semantic search)
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # Create BM25 retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(
        documents=bm25_docs,
        k=k,
        preprocess_func=_bm25_preprocess,
    )

    # Combine into hybrid retriever with equal weights
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )

    return ensemble_retriever


def search_documents(query: str) -> str:
    """
    Search the user's local document collection
    using hybrid search (BM25 + Vector).

    Combines keyword-based BM25 retrieval with
    semantic vector search for more accurate
    results on both exact terms and meaning.

    Use this tool when the user asks a question
    that requires information from the uploaded
    documents.

    Args:
        query: The user's question.

    Returns:
        Relevant passages from the documents.
    """

    retriever = _get_hybrid_retriever(k=5)

    if retriever is None:
        return "No documents have been ingested yet."

    results = retriever.invoke(query)

    if not results:
        return "No relevant information was found."

    output = []

    for doc in results:

        source = doc.metadata.get(
            "source",
            "Unknown document",
        )

        output.append(
            f"Source: {source}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(output)