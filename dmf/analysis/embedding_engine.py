"""Dense vector representations for interaction text.

This module converts raw text into fixed-length, L2-normalised numpy vectors
using FastEmbed's CPU-optimised ``TextEmbedding`` model. Unit vectors make
cosine similarity equivalent to a plain dot product, preserving an angular
measurement without magnitude bias.

Design decisions
----------------
- Lazy loading: the FastEmbed model is NOT instantiated in __init__.
  _load_model_if_needed() is called on the first get_embedding() call and
  is a no-op on every subsequent call. This avoids download and model-load
  cost until the engine is actually used.
- L2 normalisation: applied unconditionally after every embedding call.
  A zero-vector guard returns the zero vector unchanged to prevent NaN.
- Return shape: always a 1-D np.ndarray of shape (vector_dim,). FastEmbed
  yields batches; next(iter(...)) + .flatten() guarantees the 1-D contract
  regardless of FastEmbed's internal batching shape.
"""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from dmf.utils.config import VectorConfig



class EmbeddingEngine:
    """Converts text to L2-normalised dense vectors via FastEmbed.
    
        The underlying TextEmbedding model is loaded lazily on the first call
        to get_embedding(), not at construction time. Only configuration is
        stored initially so heavy I/O is deferred until the engine is used.
    
        Attributes
        ----------
        _config : VectorConfig
            Immutable configuration (model name, expected dimension, cache path).
        _model : TextEmbedding | None
            FastEmbed model instance. None until the first get_embedding() call.
    
    Args:
        config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, config: VectorConfig) -> None:
        """Store configuration; do NOT load the embedding model yet.

        Parameters
        ----------
        config : VectorConfig
            Immutable configuration for this engine instance.
        """
        self._config: VectorConfig = config
        self._model: TextEmbedding | None = None


    def get_embedding(self, text: str) -> np.ndarray:
        """Return the L2-normalised embedding vector for *text*.
        
                Triggers model loading on the first invocation; subsequent calls
                reuse the already-loaded model with no additional I/O.
        
                Parameters
                ----------
                text : str
                    Raw text to embed.
        
                Returns
                -------
                np.ndarray
                    1-D array of shape (vector_dim,) with unit L2 norm.
        
        Args:
            text: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        self._load_model_if_needed()
        raw_vector = self._generate_raw_vector(text)
        return self._apply_l2_normalization(raw_vector)


    def _load_model_if_needed(self) -> None:
        """Instantiate TextEmbedding on the first call; no-op thereafter.

        FastEmbed downloads model weights to config.cache_dir on the very
        first instantiation (~24 MB for bge-small-en-v1.5) and reads from
        the local cache on all subsequent runs.
        """
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self._config.model_name,
                cache_dir=self._config.cache_dir,
            )


    def _generate_raw_vector(self, text: str) -> np.ndarray:
        """Run FastEmbed inference and return a guaranteed 1-D raw vector.

        FastEmbed's embed() returns a generator of batches. We pass a
        single-element list, retrieve the first (and only) result with
        next(iter(...)), then call .flatten() to collapse any extra batch
        dimensions — ensuring the 1-D contract regardless of FastEmbed's
        internal output shape.

        Parameters
        ----------
        text : str
            Raw text to embed.

        Returns
        -------
        np.ndarray
            1-D array of shape (vector_dim,) before normalisation.
        """
        raw = next(iter(self._model.embed([text])))  # type: ignore[union-attr]
        return np.array(raw).flatten()

    def _apply_l2_normalization(self, vector: np.ndarray) -> np.ndarray:
        """Divide *vector* by its L2 norm to produce a unit vector.

        A zero-vector guard returns the zero vector unchanged to avoid
        producing NaN values — mathematically there is no well-defined
        unit direction for the zero vector.

        Parameters
        ----------
        vector : np.ndarray
            Raw embedding to normalise.

        Returns
        -------
        np.ndarray
            Unit-length vector (norm ≈ 1.0), or the zero vector unchanged.
        """
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm
