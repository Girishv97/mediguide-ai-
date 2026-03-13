"""
utils/rag_utils.py
RAG (Retrieval-Augmented Generation) pipeline utilities.

Responsibilities:
- Load and chunk documents (PDF, TXT, DOCX)
- Build and persist a FAISS vector store
- Retrieve top-K relevant chunks for a user query
"""

import os
import json
import logging
import hashlib
import pickle
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ─── Text Chunking ────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks by word count.

    Args:
        text: Raw text string to split.
        chunk_size: Approximate number of words per chunk.
        overlap: Number of overlapping words between consecutive chunks.

    Returns:
        List of text chunk strings.
    """
    try:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return [c for c in chunks if c.strip()]

    except Exception as exc:
        logger.error("Chunking error: %s", exc)
        raise


# ─── Document Loading ─────────────────────────────────────────────────────────

def load_document(file_path: str) -> str:
    """
    Load text content from PDF, TXT, or DOCX files.

    Args:
        file_path: Absolute or relative path to the document.

    Returns:
        Extracted plain text string.

    Raises:
        ValueError: If file format is unsupported.
        FileNotFoundError: If the file does not exist.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = os.path.splitext(file_path)[-1].lower()

        if ext == ".txt":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext == ".pdf":
            import fitz  # PyMuPDF
            text_parts = []
            doc = fitz.open(file_path)
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)

        elif ext in (".docx", ".doc"):
            import docx
            doc = docx.Document(file_path)
            return "\n".join(p.text for p in doc.paragraphs)

        else:
            raise ValueError(f"Unsupported file format: {ext}")

    except (FileNotFoundError, ValueError):
        raise
    except Exception as exc:
        logger.error("Document load error (%s): %s", file_path, exc)
        raise RuntimeError(f"Failed to load document '{file_path}': {exc}") from exc


# ─── Vector Store ─────────────────────────────────────────────────────────────

class VectorStore:
    """
    Simple in-memory FAISS vector store with optional disk persistence.

    Stores document chunks and their embeddings.
    Supports cosine-similarity retrieval of top-K chunks.
    """

    def __init__(self, store_path: Optional[str] = None):
        self.chunks: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: list[dict] = []   # {"source": ..., "chunk_index": ...}
        self.store_path = store_path
        self._file_hashes: set[str] = set()

    # ── Build ──────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        file_paths: list[str],
        chunk_size: int = 500,
        overlap: int = 50,
        embedding_provider: str = "huggingface",
    ) -> int:
        """
        Load, chunk, and embed one or more documents into the vector store.
        Skips files already indexed (by content hash).

        Returns:
            Number of new chunks added.
        """
        from models.embeddings import get_embeddings

        new_chunks: list[str] = []
        new_metadata: list[dict] = []

        for path in file_paths:
            try:
                file_hash = _file_hash(path)
                if file_hash in self._file_hashes:
                    logger.info("Skipping already-indexed file: %s", path)
                    continue

                raw_text = load_document(path)
                file_chunks = chunk_text(raw_text, chunk_size, overlap)
                source_name = os.path.basename(path)

                for idx, chunk in enumerate(file_chunks):
                    new_chunks.append(chunk)
                    new_metadata.append({"source": source_name, "chunk_index": idx})

                self._file_hashes.add(file_hash)
                logger.info("Indexed %d chunks from '%s'", len(file_chunks), source_name)

            except Exception as exc:
                logger.warning("Skipping file '%s' due to error: %s", path, exc)

        if not new_chunks:
            return 0

        new_embeddings = get_embeddings(new_chunks, provider=embedding_provider)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.chunks.extend(new_chunks)
        self.metadata.extend(new_metadata)
        return len(new_chunks)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 4,
        embedding_provider: str = "huggingface",
    ) -> list[dict]:
        """
        Retrieve the top-K most relevant chunks for a query using cosine similarity.

        Args:
            query: User's query string.
            top_k: Number of chunks to return.
            embedding_provider: Which embedding model to use.

        Returns:
            List of dicts: {"chunk": str, "source": str, "score": float}
        """
        from models.embeddings import get_embeddings

        if self.embeddings is None or len(self.chunks) == 0:
            return []

        try:
            query_vec = get_embeddings(query, provider=embedding_provider)  # (1, dim)
            query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)

            # Normalise stored embeddings
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
            normed = self.embeddings / norms

            scores = (normed @ query_vec.T).squeeze()  # (n_chunks,)
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append(
                    {
                        "chunk": self.chunks[idx],
                        "source": self.metadata[idx].get("source", "unknown"),
                        "score": float(scores[idx]),
                    }
                )
            return results

        except Exception as exc:
            logger.error("Vector search error: %s", exc)
            return []

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> None:
        """Pickle the vector store to disk."""
        save_path = path or self.store_path
        if not save_path:
            return
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self.chunks,
                        "embeddings": self.embeddings,
                        "metadata": self.metadata,
                        "file_hashes": self._file_hashes,
                    },
                    f,
                )
            logger.info("Vector store saved to %s", save_path)
        except Exception as exc:
            logger.error("Vector store save error: %s", exc)

    def load(self, path: Optional[str] = None) -> bool:
        """Load the vector store from disk. Returns True on success."""
        load_path = path or self.store_path
        if not load_path or not os.path.exists(load_path):
            return False
        try:
            with open(load_path, "rb") as f:
                data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = data["embeddings"]
            self.metadata = data["metadata"]
            self._file_hashes = data.get("file_hashes", set())
            logger.info("Vector store loaded from %s (%d chunks)", load_path, len(self.chunks))
            return True
        except Exception as exc:
            logger.error("Vector store load error: %s", exc)
            return False

    def clear(self) -> None:
        """Reset the vector store."""
        self.chunks = []
        self.embeddings = None
        self.metadata = []
        self._file_hashes = set()

    def __len__(self) -> int:
        return len(self.chunks)


# ─── Context Builder ──────────────────────────────────────────────────────────

def build_rag_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a structured context string for the LLM prompt.

    Args:
        results: Output from VectorStore.search().

    Returns:
        Formatted context block as a string.
    """
    if not results:
        return ""

    context_lines = ["### Relevant Knowledge Base Excerpts\n"]
    for i, r in enumerate(results, 1):
        context_lines.append(
            f"**[{i}] Source: {r['source']}** (relevance: {r['score']:.2f})\n{r['chunk']}\n"
        )
    return "\n".join(context_lines)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _file_hash(file_path: str) -> str:
    """Compute MD5 hash of file contents for deduplication."""
    h = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()
