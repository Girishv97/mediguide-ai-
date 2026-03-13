"""
models/embeddings.py
Embedding model wrappers for RAG (Retrieval-Augmented Generation).
Supports HuggingFace (local/free) and OpenAI embeddings.
"""

import logging
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)


# ─── HuggingFace / SentenceTransformers ──────────────────────────────────────

_hf_model_cache: dict = {}


def get_huggingface_embeddings(
    texts: Union[str, list[str]],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Generate embeddings using a local HuggingFace SentenceTransformer model.
    Models are cached in memory after first load to avoid repeated disk reads.

    Args:
        texts: A single string or list of strings to embed.
        model_name: HuggingFace model identifier.

    Returns:
        NumPy array of shape (n_texts, embedding_dim).
    """
    try:
        from sentence_transformers import SentenceTransformer

        if isinstance(texts, str):
            texts = [texts]

        if model_name not in _hf_model_cache:
            logger.info("Loading embedding model: %s", model_name)
            _hf_model_cache[model_name] = SentenceTransformer(model_name)

        model = _hf_model_cache[model_name]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings

    except Exception as exc:
        logger.error("HuggingFace embedding error: %s", exc)
        raise RuntimeError(f"Embedding error (HuggingFace): {exc}") from exc


# ─── OpenAI Embeddings ────────────────────────────────────────────────────────

def get_openai_embeddings(
    texts: Union[str, list[str]],
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
) -> np.ndarray:
    """
    Generate embeddings using the OpenAI Embeddings API.

    Args:
        texts: A single string or list of strings.
        model: OpenAI embedding model name.
        api_key: Optional API key override.

    Returns:
        NumPy array of shape (n_texts, embedding_dim).
    """
    try:
        from openai import OpenAI
        from config.config import OPENAI_API_KEY

        if isinstance(texts, str):
            texts = [texts]

        client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model=model)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    except Exception as exc:
        logger.error("OpenAI embedding error: %s", exc)
        raise RuntimeError(f"Embedding error (OpenAI): {exc}") from exc


# ─── Unified Dispatcher ───────────────────────────────────────────────────────

def get_embeddings(
    texts: Union[str, list[str]],
    provider: str = "huggingface",
    model: str | None = None,
) -> np.ndarray:
    """
    Unified embedding interface.

    Args:
        texts: Text(s) to embed.
        provider: "huggingface" | "openai"
        model: Optional model name override.

    Returns:
        NumPy embedding array.
    """
    from config.config import (
        HUGGINGFACE_EMBEDDING_MODEL,
        OPENAI_EMBEDDING_MODEL,
    )

    provider = provider.lower().strip()

    if provider == "huggingface":
        return get_huggingface_embeddings(
            texts, model_name=model or HUGGINGFACE_EMBEDDING_MODEL
        )
    elif provider == "openai":
        return get_openai_embeddings(
            texts, model=model or OPENAI_EMBEDDING_MODEL
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider '{provider}'. Choose: huggingface, openai."
        )
