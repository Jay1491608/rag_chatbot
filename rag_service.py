"""Lazy, process-local RAG service used by the FastAPI application."""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from config import API_KEY, BASE_URL, MODEL_NAME


class RAGService:
    """Build and cache the RAG pipeline on the first question."""

    def __init__(self) -> None:
        self._project_dir = Path(__file__).parent
        self._index_dir = self._project_dir / "faiss_index"
        self._initialize_lock = Lock()
        self._qa_chain: Any = None

    def _initialize(self) -> None:
        if self._qa_chain is not None:
            return

        with self._initialize_lock:
            if self._qa_chain is not None:
                return

            if not self._index_dir.exists():
                raise RuntimeError(
                    "FAISS index is missing. Run 'python build_index.py' "
                    "before deploying."
                )

            from langchain_community.vectorstores import FAISS
            from langchain_classic.chains import create_retrieval_chain
            from langchain_classic.chains.combine_documents import (
                create_stuff_documents_chain,
            )
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_openai import ChatOpenAI

            print("Loading embedding model...", flush=True)
            embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            print("Loading FAISS index...", flush=True)
            vector_db = FAISS.load_local(
                str(self._index_dir),
                embedding_model,
                allow_dangerous_deserialization=True,
            )

            print("Creating retriever...", flush=True)
            retriever = vector_db.as_retriever(search_kwargs={"k": 2})

            print("Creating LLM...", flush=True)
            llm = ChatOpenAI(
                model=MODEL_NAME,
                api_key=API_KEY,
                base_url=BASE_URL,
            )

            prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an assistant for question-answering tasks.\n"
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, just say "
                    "that you don't know.\n\nContext:\n{context}",
                ),
                ("human", "{input}"),
            ])
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            self._qa_chain = create_retrieval_chain(
                retriever, question_answer_chain
            )
            print("RAG initialized successfully.", flush=True)

    def ask(self, question: str) -> str:
        self._initialize()
        result = self._qa_chain.invoke({"input": question})
        return result["answer"]


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Return the one RAG service instance for this server process."""
    return RAGService()