"""Local embedding provider using sentence-transformers.

Uses BAAI/bge-small-en-v1.5 by default - a high-quality, compact model
that produces 384-dimensional embeddings with excellent semantic similarity.

Configuration via environment variables:
- VALENCE_EMBEDDING_MODEL_PATH: Model name or local path (default: BAAI/bge-small-en-v1.5)
- VALENCE_EMBEDDING_DEVICE: Device to use (cpu|cuda, default: cpu)

The model is lazily loaded on first use and cached for subsequent calls.

For offline/air-gapped environments:
- Pre-download model: python scripts/download_model.py
- Or set VALENCE_EMBEDDING_MODEL_PATH to a local filesystem path
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Model singleton (lazy loaded)
_model: "SentenceTransformer | None" = None
_model_lock = threading.Lock()  # Thread lock for model initialization

# Default model - bge-small-en-v1.5 is excellent for semantic similarity
# 384 dimensions, ~33M params, fast inference
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models produce 384-dimensional embeddings
EMBEDDING_DIMENSIONS = 384


class ModelLoadError(Exception):
    """Raised when the embedding model cannot be loaded."""

    pass


def _is_local_path(model_path: str) -> bool:
    """Check if model_path is a local filesystem path vs HuggingFace model name."""
    # Local paths contain path separators or start with . or /
    path = Path(model_path)
    return path.exists() or model_path.startswith(("/", "./", "../", "~"))


def get_model() -> "SentenceTransformer":
    """Get or initialize the sentence transformer model.
    
    The model is lazily loaded and cached for reuse.
    Thread-safe initialization using double-checked locking pattern.
    
    Environment variables:
        VALENCE_EMBEDDING_MODEL_PATH: Model name/path (default: BAAI/bge-small-en-v1.5)
            - HuggingFace model name: downloads if not cached
            - Local filesystem path: loads directly (for offline use)
        VALENCE_EMBEDDING_DEVICE: Device to run on (cpu|cuda, default: cpu)
    
    Returns:
        Loaded SentenceTransformer model
        
    Raises:
        ModelLoadError: If model cannot be loaded (with helpful offline instructions)
    """
    global _model
    
    if _model is None:
        with _model_lock:
            # Double-check after acquiring lock
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise ModelLoadError(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    ) from e
                
                from ...core.config import get_config
                config = get_config()
                model_path = config.embedding_model_path
                device = config.embedding_device
                
                is_local = _is_local_path(model_path)
                
                if is_local:
                    resolved_path = Path(model_path).expanduser().resolve()
                    if not resolved_path.exists():
                        raise ModelLoadError(
                            f"Local model path not found: {resolved_path}\n\n"
                            "To download the model for offline use:\n"
                            "  python scripts/download_model.py --save-path /path/to/model\n\n"
                            "Then set:\n"
                            "  export VALENCE_EMBEDDING_MODEL_PATH=/path/to/model"
                        )
                    logger.info(f"Loading embedding model from local path: {resolved_path}")
                    load_path = str(resolved_path)
                else:
                    logger.info(f"Loading embedding model: {model_path} (device={device})")
                    load_path = model_path
                
                try:
                    _model = SentenceTransformer(load_path, device=device)
                except OSError as e:
                    # Common error when offline and model not cached
                    error_msg = str(e).lower()
                    if "connection" in error_msg or "resolve" in error_msg or "network" in error_msg:
                        raise ModelLoadError(
                            f"Cannot download model '{model_path}' - network unavailable.\n\n"
                            "For offline/air-gapped environments:\n\n"
                            "Option 1: Pre-download on a connected machine:\n"
                            "  python scripts/download_model.py\n"
                            "  scp -r ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5 target:~/.cache/huggingface/hub/\n\n"
                            "Option 2: Save to custom path:\n"
                            "  python scripts/download_model.py --save-path /opt/valence/models/bge-small-en-v1.5\n"
                            "  # Then on target:\n"
                            "  export VALENCE_EMBEDDING_MODEL_PATH=/opt/valence/models/bge-small-en-v1.5"
                        ) from e
                    raise ModelLoadError(f"Failed to load embedding model: {e}") from e
                except Exception as e:
                    raise ModelLoadError(f"Failed to load embedding model '{model_path}': {e}") from e
                
                dim = _model.get_sentence_embedding_dimension()
                logger.info(f"Embedding model ready (dim={dim})")
    
    return _model


def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding for text.
    
    Uses BGE model which produces L2-normalized embeddings by default
    when normalize_embeddings=True.
    
    Args:
        text: Text to embed
        
    Returns:
        384-dimensional embedding vector (L2 normalized)
    """
    model = get_model()
    
    # BGE models work best with a query prefix for retrieval tasks,
    # but for storage we embed without prefix
    embedding = model.encode(text, normalize_embeddings=True)
    
    return embedding.tolist()


def generate_embeddings_batch(
    texts: list[str],
    batch_size: int = 32,
    show_progress: bool | None = None,
) -> list[list[float]]:
    """Generate embeddings for multiple texts efficiently.
    
    Uses batching for better throughput on GPU/CPU.
    Shows progress bar for large batches (>100 texts).
    
    Args:
        texts: List of texts to embed
        batch_size: Batch size for processing (default: 32)
        show_progress: Show progress bar (default: True if >100 texts)
        
    Returns:
        List of 384-dimensional embedding vectors (L2 normalized)
    """
    model = get_model()
    
    if show_progress is None:
        show_progress = len(texts) > 100
    
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=show_progress,
    )
    
    return embeddings.tolist()


def reset_model() -> None:
    """Reset the cached model (useful for testing).
    
    Thread-safe reset using lock.
    """
    global _model
    with _model_lock:
        _model = None
