import sys
from pathlib import Path
from app.tools.document_ingestion import ingest_pdf

DOCUMENTS_DIR = Path("documents")


def main():
    if not DOCUMENTS_DIR.exists():
        print(f"Error: Directory '{DOCUMENTS_DIR}' does not exist.")
        sys.exit(1)

    # Find all PDF files (case-insensitive) in the documents directory
    pdf_files = [
        f for f in DOCUMENTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ]

    if not pdf_files:
        print(f"No PDF files found in '{DOCUMENTS_DIR}'. Please place your PDF documents there.")
        return

    print(f"Found {len(pdf_files)} PDF file(s) in '{DOCUMENTS_DIR}':")
    for pdf in pdf_files:
        print(f"  • {pdf.name}")
    print()

    total_chunks = 0
    successful_docs = 0

    for pdf_path in pdf_files:
        print(f"=== Processing: {pdf_path.name} ===")
        try:
            count = ingest_pdf(str(pdf_path))
            total_chunks += count
            successful_docs += 1
            print(f"Done: {count} chunk(s) ingested.\n")
        except Exception as e:
            print(f"Error ingesting {pdf_path.name}: {e}\n")

    print(
        f"Completed: Ingested {total_chunks} total chunk(s) "
        f"across {successful_docs}/{len(pdf_files)} PDF(s)."
    )


if __name__ == "__main__":
    main()