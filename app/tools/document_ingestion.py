import os
import uuid

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader


load_dotenv()


CHROMA_DB_PATH = "chroma_db"
COLLECTION_NAME = "documents"


embeddings = OpenAIEmbeddings(
    model=os.getenv("LM_STUDIO_EMBEDDING_MODEL"),
    base_url=os.getenv("LM_STUDIO_BASE_URL"),
    api_key=os.getenv("LM_STUDIO_API_KEY"),
    check_embedding_ctx_length=False,
)


def split_text(text):
    """
    Split text into chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Uses paragraph, line, word, and character
    boundaries to create semantically coherent
    chunks of up to 500 characters with 50
    character overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    print(f"Original length: {len(text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(
        f"Chunk sizes: {[len(c) for c in chunks]}"
    )

    if chunks:
        print(
            f"\nFirst chunk preview:\n"
            f"{chunks[0][:200]}..."
        )

    return chunks


def ingest_pdf(file_path):
    """
    Ingest a PDF file into the Chroma vectorstore.

    Extracts text from all pages, splits it using
    RecursiveCharacterTextSplitter, and stores the
    chunks with metadata in ChromaDB via LangChain's
    Chroma wrapper.
    """

    reader = PdfReader(file_path)

    full_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            full_text += text + "\n"

    chunks = split_text(full_text)

    source_name = os.path.basename(file_path)

    # Create LangChain Document objects with metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={
                "source": source_name,
                "chunk_id": str(uuid.uuid4()),
            },
        )
        for chunk in chunks
    ]

    if not documents:
        print(f"No text could be extracted from '{source_name}'. Skipping.")
        return 0

    # Store via LangChain's Chroma wrapper
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB_PATH,
    )

    print(
        f"Ingested {len(documents)} chunks "
        f"from '{source_name}'"
    )

    return len(documents)