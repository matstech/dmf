"""
tests/test_geometry.py
----------------------
Unit tests for dmf/core/geometry.py — InteractionMatrix, calculate_centroid,
and calculate_divergence.

All tests use small hand-crafted vectors (not FastEmbed) so they run
instantly with no model download. This validates the pure geometry logic
independently of the embedding pipeline.

Coverage:
  calculate_centroid
    - Correct centroid for 3 known vectors.
    - Single-vector centroid equals the vector itself.

  calculate_divergence
    - Identical vectors → divergence = 0.
    - Orthogonal unit vectors → divergence = 1.
    - Opposite vectors → divergence = 2.
    - Zero vector edge case → divergence = 0 (guard).

  InteractionMatrix
    - First interaction returns divergence 0.0 (empty window edge case).
    - Similar vectors produce low divergence.
    - Out-of-context vector produces higher divergence.
    - Sliding window correctly evicts oldest vectors (FIFO).
    - Window size property reflects config.
"""

from __future__ import annotations

import numpy as np
import pytest

from dmf.analysis.geometry import (
    InteractionMatrix,
    calculate_centroid,
    calculate_divergence,
)
from dmf.utils.config import VectorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> np.ndarray:
    """Return the L2-normalised version of *v* as a 1-D numpy array."""
    a = np.array(v, dtype=np.float64)
    return a / np.linalg.norm(a)


# ---------------------------------------------------------------------------
# calculate_centroid
# ---------------------------------------------------------------------------

class TestCalculateCentroid:
    """Tests for the pure calculate_centroid function."""

    def test_centroid_of_three_vectors_is_element_wise_mean(self) -> None:
        """Centroid of [1,0,0], [0,1,0], [0,0,1] must be [1/3, 1/3, 1/3]."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])

        centroid = calculate_centroid([v1, v2, v3])
        expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])

        np.testing.assert_array_almost_equal(centroid, expected)

    def test_centroid_of_single_vector_is_itself(self) -> None:
        """The mean of one vector is the vector itself."""
        v = np.array([0.5, 0.3, 0.8])
        centroid = calculate_centroid([v])
        np.testing.assert_array_almost_equal(centroid, v)


# ---------------------------------------------------------------------------
# calculate_divergence
# ---------------------------------------------------------------------------

class TestCalculateDivergence:
    """Tests for the pure calculate_divergence function."""

    def test_divergence_is_zero_for_identical_vectors(self) -> None:
        """D = 1 − cos(0°) = 0 when both vectors point in the same direction."""
        v = _unit([1.0, 0.0, 0.0])
        assert abs(calculate_divergence(v, v)) < 1e-6

    def test_divergence_is_one_for_orthogonal_vectors(self) -> None:
        """D = 1 − cos(90°) = 1 for perpendicular unit vectors."""
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = _unit([0.0, 1.0, 0.0])
        assert abs(calculate_divergence(v1, v2) - 1.0) < 1e-6

    def test_divergence_is_two_for_opposite_vectors(self) -> None:
        """D = 1 − cos(180°) = 2 for diametrically opposed unit vectors."""
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = _unit([-1.0, 0.0, 0.0])
        assert abs(calculate_divergence(v1, v2) - 2.0) < 1e-6

    def test_divergence_returns_zero_for_zero_vector(self) -> None:
        """Zero-vector guard: undefined angle → return 0.0 (stable default)."""
        v = _unit([1.0, 0.0, 0.0])
        zero = np.array([0.0, 0.0, 0.0])
        assert calculate_divergence(v, zero) == 0.0
        assert calculate_divergence(zero, v) == 0.0

    def test_divergence_increases_for_out_of_context_vector(self) -> None:
        """A vector far from the centroid must have higher D than a nearby one."""
        centroid = _unit([1.0, 0.0, 0.0])

        nearby = _unit([0.95, 0.05, 0.0])
        faraway = _unit([0.0, 0.0, 1.0])

        d_nearby = calculate_divergence(nearby, centroid)
        d_faraway = calculate_divergence(faraway, centroid)

        assert d_faraway > d_nearby


# ---------------------------------------------------------------------------
# InteractionMatrix
# ---------------------------------------------------------------------------

class TestInteractionMatrix:
    """Tests for the InteractionMatrix sliding-window container."""

    def test_first_interaction_returns_zero_divergence(self) -> None:
        """Empty window → no prior context → D must be 0.0 (SPEC §4 edge case)."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        d = matrix.add_vector(v)
        assert d == 0.0

    def test_identical_subsequent_vector_has_zero_divergence(self) -> None:
        """Adding the same vector twice: D ≈ 0 (perfectly aligned with centroid)."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v)  # first — D=0.0 by edge-case rule
        d = matrix.add_vector(v)  # second — centroid IS v, new IS v
        assert abs(d) < 1e-6

    def test_out_of_context_vector_has_higher_divergence(self) -> None:
        """A semantically distant vector must produce a larger D than a close one."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=10))

        # Fill the window with similar vectors along the x-axis
        for _ in range(3):
            matrix.add_vector(_unit([1.0, 0.0, 0.0]))

        # Similar vector: small angle from centroid
        d_similar = matrix.add_vector(_unit([0.95, 0.05, 0.0]))

        # Reset for fair comparison
        matrix2 = InteractionMatrix(config=VectorConfig(window_size=10))
        for _ in range(3):
            matrix2.add_vector(_unit([1.0, 0.0, 0.0]))

        # Out-of-context vector: large angle from centroid
        d_outlier = matrix2.add_vector(_unit([0.0, 0.0, 1.0]))

        assert d_outlier > d_similar

    def test_sliding_window_evicts_oldest_vectors(self) -> None:
        """When window_size=3, adding a 4th vector must evict the 1st (FIFO)."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=3))

        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([0.0, 0.0, 1.0])
        v4 = np.array([0.5, 0.5, 0.0])

        matrix.add_vector(v1)
        matrix.add_vector(v2)
        matrix.add_vector(v3)
        assert matrix.size == 3

        matrix.add_vector(v4)
        assert matrix.size == 3  # still 3 — v1 was evicted

        # Centroid should now be the L2-normalised mean of v2, v3, v4 (v1 gone)
        raw_mean = np.mean(np.stack([v2, v3, v4]), axis=0)
        expected_centroid = raw_mean / np.linalg.norm(raw_mean)
        actual_centroid = matrix.get_centroid()
        assert actual_centroid is not None
        np.testing.assert_array_almost_equal(actual_centroid, expected_centroid)

    def test_window_size_property_reflects_config(self) -> None:
        """window_size property must match the value from VectorConfig."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=7))
        assert matrix.window_size == 7

    def test_is_empty_before_any_vector_added(self) -> None:
        """A freshly created matrix must report is_empty=True."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        assert matrix.is_empty is True
        assert matrix.size == 0

    def test_get_centroid_returns_none_when_empty(self) -> None:
        """get_centroid() must return None when no vectors have been added."""
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        assert matrix.get_centroid() is None

    def test_centroid_is_l2_normalised(self) -> None:
        """The cached centroid must always have unit L2 norm.

        The raw mean of non-collinear unit vectors is shorter than 1.
        InteractionMatrix must project it back onto the unit hypersphere.
        """
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        matrix.add_vector(_unit([1.0, 0.0, 0.0]))
        matrix.add_vector(_unit([0.0, 1.0, 0.0]))
        matrix.add_vector(_unit([0.0, 0.0, 1.0]))

        centroid = matrix.get_centroid()
        assert centroid is not None
        assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-6

    def test_centroid_is_o1_read(self) -> None:
        """get_centroid() must return the cached attribute directly, not recompute.

        We verify this by checking that the returned object *is* the same
        object as the private attribute (identity, not equality).
        """
        matrix = InteractionMatrix(config=VectorConfig(window_size=5))
        matrix.add_vector(_unit([1.0, 0.0, 0.0]))
        assert matrix.get_centroid() is matrix._current_centroid


# ---------------------------------------------------------------------------
# InteractionMatrix — remove_vector
# ---------------------------------------------------------------------------

class TestRemoveVector:
    """remove_vector must remove by object identity, not value equality."""

    def test_remove_existing_vector_returns_true(self) -> None:
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v)
        assert matrix.remove_vector(v) is True

    def test_remove_existing_vector_decreases_size(self) -> None:
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v)
        matrix.remove_vector(v)
        assert matrix.size == 0

    def test_remove_absent_vector_returns_false(self) -> None:
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v_in = _unit([1.0, 0.0, 0.0])
        v_out = _unit([0.0, 1.0, 0.0])
        matrix.add_vector(v_in)
        assert matrix.remove_vector(v_out) is False
        assert matrix.size == 1

    def test_remove_from_empty_matrix_returns_false(self) -> None:
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        assert matrix.remove_vector(v) is False

    def test_centroid_reset_to_none_after_removing_sole_vector(self) -> None:
        """When the last vector is removed the centroid must become None."""
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v)
        matrix.remove_vector(v)
        assert matrix.get_centroid() is None

    def test_centroid_rebuilt_after_partial_removal(self) -> None:
        """After removing one vector, centroid is recomputed from survivors."""
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = _unit([0.0, 1.0, 0.0])
        matrix.add_vector(v1)
        matrix.add_vector(v2)
        matrix.remove_vector(v1)
        # Only v2 remains; centroid should be unit-v2 (already unit length)
        centroid = matrix.get_centroid()
        assert centroid is not None
        assert matrix.size == 1
        assert abs(float(np.linalg.norm(centroid)) - 1.0) < 1e-6

    def test_uses_object_identity_not_value_equality(self) -> None:
        """Two arrays with the same values but different identities:
        only the inserted object should be removable."""
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v1 = _unit([1.0, 0.0, 0.0])
        v2 = v1.copy()  # identical values, different Python object
        matrix.add_vector(v1)
        # v2 is not v1 → should not be found
        assert matrix.remove_vector(v2) is False
        assert matrix.size == 1
        # v1 is v1 → should be found
        assert matrix.remove_vector(v1) is True
        assert matrix.size == 0

    def test_remove_only_first_occurrence_of_duplicate_object(self) -> None:
        """If somehow the same object is added twice, only one should be removed."""
        matrix = InteractionMatrix(VectorConfig(window_size=5))
        v = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v)
        matrix.add_vector(v)
        matrix.remove_vector(v)
        # One occurrence removed, one remains
        assert matrix.size == 1

    def test_scrolled_out_vector_not_in_window(self) -> None:
        """A vector evicted by the FIFO window cannot be found for removal."""
        matrix = InteractionMatrix(VectorConfig(window_size=2))
        v_old = _unit([1.0, 0.0, 0.0])
        matrix.add_vector(v_old)
        # Fill the window past capacity — v_old is scrolled out
        matrix.add_vector(_unit([0.0, 1.0, 0.0]))
        matrix.add_vector(_unit([0.0, 0.0, 1.0]))
        # v_old is no longer in the deque
        assert matrix.remove_vector(v_old) is False
