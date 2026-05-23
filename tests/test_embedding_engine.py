"""
tests/test_embedding_engine.py
-------------------------------
Unit tests for `dmf.analysis.embedding_engine.EmbeddingEngine`.

FastEmbed downloads bge-small-en-v1.5 (~24 MB) on the first run and caches
it locally under models/embeddings/. Subsequent runs read from the cache and
are fast. The module-scoped fixture loads the engine once for the entire
session to avoid repeated model-load overhead across tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmf.analysis.embedding_engine import EmbeddingEngine
from dmf.utils.config import VectorConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> EmbeddingEngine:
    """Module-scoped EmbeddingEngine instance.

    FastEmbed loads/downloads the model on the first get_embedding() call
    inside this fixture. All tests in this module reuse the same instance,
    so the model is loaded exactly once per test session.
    """
    return EmbeddingEngine(config=VectorConfig())


@pytest.fixture(scope="module")
def sample_embedding(engine: EmbeddingEngine) -> np.ndarray:
    """A single embedding reused across multiple shape/norm tests."""
    return engine.get_embedding("The quick brown fox jumps over the lazy dog.")


# ---------------------------------------------------------------------------
# Lazy-loading tests (use a FRESH engine — independent of the module fixture)
# ---------------------------------------------------------------------------

def test_model_is_not_loaded_before_first_call() -> None:
    """_model must be None immediately after construction (lazy sentinel)."""
    fresh_engine = EmbeddingEngine(config=VectorConfig())
    assert fresh_engine._model is None


def test_model_is_loaded_after_first_call() -> None:
    """_model must be set after the first get_embedding() call."""
    fresh_engine = EmbeddingEngine(config=VectorConfig())
    assert fresh_engine._model is None          # pre-condition
    fresh_engine.get_embedding("hello")
    assert fresh_engine._model is not None      # post-condition


# ---------------------------------------------------------------------------
# Output type and shape tests
# ---------------------------------------------------------------------------

def test_output_is_numpy_ndarray(sample_embedding: np.ndarray) -> None:
    """get_embedding() must return a numpy ndarray (type contract)."""
    assert isinstance(sample_embedding, np.ndarray)


def test_embedding_has_correct_dimension(sample_embedding: np.ndarray) -> None:
    """Output shape must be (384,) — native bge-small-en-v1.5 dimension."""
    config = VectorConfig()
    assert sample_embedding.shape == (config.vector_dim,)


# ---------------------------------------------------------------------------
# L2 normalisation tests
# ---------------------------------------------------------------------------

def test_embedding_is_l2_normalized(sample_embedding: np.ndarray) -> None:
    """L2 norm of every output vector must be within 1e-6 of 1.0."""
    norm = float(np.linalg.norm(sample_embedding))
    assert abs(norm - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------

def test_identical_texts_have_cosine_similarity_of_one(
    engine: EmbeddingEngine,
) -> None:
    """Cosine similarity of a unit vector with itself must be ≈ 1.0.

    Because vectors are L2-normalised, cosine similarity equals the dot
    product. dot(v, v) == ||v||² == 1.0 for unit vectors.
    """
    text = "Artificial intelligence is transforming the world."
    vec = engine.get_embedding(text)
    cosine = float(np.dot(vec, vec))
    assert abs(cosine - 1.0) < 1e-6


def test_similar_texts_have_high_cosine_similarity(
    engine: EmbeddingEngine,
) -> None:
    """Semantically similar sentences should have cosine similarity > 0.9."""
    v1 = engine.get_embedding("The cat sat on the mat.")
    v2 = engine.get_embedding("A cat is sitting on a mat.")
    cosine = float(np.dot(v1, v2))
    assert cosine > 0.9


def test_dissimilar_texts_have_lower_cosine_similarity_than_similar(
    engine: EmbeddingEngine,
) -> None:
    """Semantically distant pairs must score lower than semantically close pairs.

    Validates the relative ordering of cosine similarity — not absolute values.
    """
    similar_1 = engine.get_embedding("The cat sat on the mat.")
    similar_2 = engine.get_embedding("A cat is sitting on a mat.")
    dissimilar = engine.get_embedding(
        "Quantum computing leverages superposition to solve NP-hard problems."
    )

    cosine_similar = float(np.dot(similar_1, similar_2))
    cosine_dissimilar = float(np.dot(similar_1, dissimilar))

    assert cosine_dissimilar < cosine_similar
