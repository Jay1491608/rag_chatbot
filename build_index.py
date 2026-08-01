"""Build the FAISS index used by the production RAG service.

Run this once from the project directory before deploying:
    python build_index.py
"""

from pathlib import Path


def main() -> None:
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    project_dir = Path(__file__).parent
    source_path = project_dir / "system_design.txt"
    index_dir = project_dir / "faiss_index"

    print("Reading system_design.txt...", flush=True)
    text = source_path.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    docs = splitter.create_documents([text])

    print("Loading embedding model...", flush=True)
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building FAISS index...", flush=True)
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local(str(index_dir))
    print(f"FAISS index saved to {index_dir}", flush=True)


if __name__ == "__main__":
    main()