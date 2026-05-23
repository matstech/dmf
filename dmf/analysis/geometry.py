"""Semantic geometry for sliding-window divergence tracking.

The InteractionMatrix maintains a fixed-size window of the most recent
interaction embedding vectors. On each new interaction it:

  1. Computes the centroid (element-wise mean) of the current window.
  2. Measures semantic divergence D = 1 − cosine_similarity(new, centroid).
  3. Appends the new vector, evicting the oldest if the window is full.

D = 0 means the new interaction is perfectly aligned with the recent
conversation context; D → 2 means maximal opposition. The first
interaction in an empty window always returns D = 0.0 because there is
no prior context to diverge from.

Module-level pure functions (calculate_centroid, calculate_divergence)
are exposed for direct unit testing and can be reused outside the
InteractionMatrix when needed.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from dmf.utils.config import VectorConfig
from dmf.utils.config_loader import DMFConfig



def calculate_centroid(vectors: Sequence[np.ndarray]) -> np.ndarray:
    """Return the element-wise mean (centroid) of a sequence of vectors.
    
        Parameters
        ----------
        vectors : Sequence[np.ndarray]
            One or more vectors of identical shape. Must not be empty —
            callers are responsible for the empty-check before calling.
    
        Returns
        -------
        np.ndarray
            1-D array whose shape matches the input vectors.
    
    Args:
        vectors: See the function signature and surrounding type hints.
    
    Raises:
        None.
    """
    return np.mean(np.stack(vectors), axis=0)


def calculate_divergence(
    new_vector: np.ndarray,
    centroid: np.ndarray,
) -> float:
    """Return D = 1 − cosine_similarity(new_vector, centroid).
    
        Cosine similarity is computed with the full formula:
    
            cos(θ) = dot(a, b) / (‖a‖ · ‖b‖)
    
        This is safe even when the centroid is not unit-length (the mean of
        unit vectors is generally shorter than 1). The result is clipped to
        [−1, 1] to guard against floating-point drift before subtracting
        from 1.
    
        Edge cases
        ----------
        - If either vector has zero norm, the angle is undefined and the
          function returns 0.0 (no divergence) to keep downstream scoring
          numerically stable.
    
        Parameters
        ----------
        new_vector : np.ndarray
            The interaction vector to compare against the context centroid.
        centroid : np.ndarray
            The mean of the current sliding window vectors.
    
        Returns
        -------
        float
            Divergence score in [0.0, 2.0]. 0 = identical direction,
            1 = orthogonal, 2 = diametrically opposed.
    
    Args:
        new_vector: See the function signature and surrounding type hints.
        centroid: See the function signature and surrounding type hints.
    
    Raises:
        None.
    """
    norm_new = float(np.linalg.norm(new_vector))
    norm_centroid = float(np.linalg.norm(centroid))

    if norm_new == 0.0 or norm_centroid == 0.0:
        return 0.0

    cosine_sim = float(np.dot(new_vector, centroid) / (norm_new * norm_centroid))
    cosine_sim = float(np.clip(cosine_sim, -1.0, 1.0))
    return 1.0 - cosine_sim



class InteractionMatrix:
    """Sliding-window container for recent interaction embedding vectors.
    
        Backed by a ``collections.deque`` with ``maxlen=config.window_size``.
        When the window is full, appending a new vector automatically evicts
        the oldest entry (FIFO) — no manual bookkeeping required.
    
        The centroid is maintained as a **stateful, L2-normalised** attribute
        (``_current_centroid``) that is recomputed only inside ``add_vector``
        after the deque is updated. ``get_centroid()`` is therefore an O(1)
        read with no recomputation. L2 normalisation projects the mean vector
        back onto the unit hypersphere, ensuring that divergence comparisons
        remain pure angular measurements.
    
        Attributes
        ----------
        _config : VectorConfig
            Immutable configuration (only ``window_size`` is used here).
        _vectors : deque[np.ndarray]
            Bounded deque of the most recent embedding vectors.
        _current_centroid : np.ndarray | None
            Cached, L2-normalised centroid of the current window. None when
            the window is empty.
    
    Args:
        config: See the function signature and surrounding type hints.
    
    Returns:
        Instance of this class.
    
    Raises:
        None.
    """

    def __init__(self, config: VectorConfig) -> None:
        """Initialise the matrix with an empty sliding window.

        Parameters
        ----------
        config : VectorConfig
            Immutable configuration. ``window_size`` controls the maximum
            number of vectors retained.
        """
        self._config: VectorConfig = config
        self._vectors: deque[np.ndarray] = deque(maxlen=config.window_size)
        self._current_centroid: np.ndarray | None = None


    @classmethod
    def from_dmf_config(cls, config: DMFConfig) -> InteractionMatrix:
        """Construct an ``InteractionMatrix`` from a universal ``DMFConfig``.
        
                Uses ``nlp.vector_dim`` and ``capacity.window_size`` from the TOML.
        
                Translation map
                ---------------
                ==================== ====================== ===================
                TOML key             DMFConfig path         VectorConfig field
                ==================== ====================== ===================
                nlp.vector_dim       config.nlp.vector_dim  vector_dim
                capacity.window_size config.capacity.window_size window_size
                ==================== ====================== ===================
        
                Parameters
                ----------
                config : DMFConfig
                    Fully populated config object returned by ``load_dmf_config()``.
        
                Returns
                -------
                InteractionMatrix
                    Fully initialised instance with an empty sliding window.
        
        Args:
            config: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        vector_cfg = VectorConfig(
            vector_dim=config.nlp.vector_dim,
            window_size=config.capacity.window_size,
        )
        return cls(config=vector_cfg)


    def add_vector(self, vector: np.ndarray) -> float:
        """Append *vector* to the window and return its divergence score.
        
                Execution order (critical for correctness):
        
                1. Compute divergence D against the **previous** centroid — this
                   measures how far the new interaction departs from the existing
                   context *before* it becomes part of that context.
                2. Append the new vector to the deque (may evict the oldest).
                3. Recompute and L2-normalise the centroid from the updated window.
        
                On the very first call (empty window) divergence is defined as
                0.0 because there is no prior context to diverge from.
        
                Parameters
                ----------
                vector : np.ndarray
                    L2-normalised embedding vector for the new interaction.
        
                Returns
                -------
                float
                    Semantic divergence D ∈ [0.0, 2.0].
        
        Args:
            vector: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        if len(self._vectors) == 0:
            self._vectors.append(vector)
            self._recalculate_centroid()
            return 0.0

        # Divergence must be measured before the window absorbs the new vector;
        # otherwise every turn partially compares against itself.
        divergence = calculate_divergence(vector, self._current_centroid)

        self._vectors.append(vector)

        self._recalculate_centroid()

        return divergence

    def get_centroid(self) -> np.ndarray | None:
        """Return the cached, L2-normalised centroid (O(1) read).
        
                Returns
                -------
                np.ndarray | None
                    Unit-length centroid of the current window, or None when the
                    window is empty.
        
        Raises:
            None.
        """
        return self._current_centroid


    def _recalculate_centroid(self) -> None:
        """Recompute the centroid from the current deque and L2-normalise it.

        The raw element-wise mean of the window vectors is divided by its
        L2 norm to project it back onto the unit hypersphere. This ensures
        that ``calculate_divergence`` always compares unit vectors, making
        D a pure angular measurement with no magnitude bias.

        A zero-norm guard preserves the zero vector unchanged to avoid NaN.
        """
        mean = calculate_centroid(list(self._vectors))
        norm = float(np.linalg.norm(mean))
        if norm > 0.0:
            self._current_centroid = mean / norm
        else:
            self._current_centroid = mean


    @property
    def size(self) -> int:
        """Number of vectors currently in the sliding window.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return len(self._vectors)

    @property
    def is_empty(self) -> bool:
        """True when the window contains no vectors.
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return len(self._vectors) == 0

    @property
    def window_size(self) -> int:
        """Maximum capacity of the sliding window (from config).
        
        Returns:
            See the return type annotation.
        
        Raises:
            None.
        """
        return self._config.window_size

    def remove_vector(self, vector: np.ndarray) -> bool:
        """Remove a specific vector from the sliding window by object identity.
        
                Comparison uses Python's ``is`` operator (object identity), not
                element-wise numpy equality.  This is safe because the same
                ``np.ndarray`` object is stored on both the ``MemoryEntry`` and
                this deque — ``add_vector`` receives the reference and appends it
                directly without copying.
        
                If the vector is no longer in the window (evicted by FIFO overflow
                because more than ``window_size`` interactions arrived since it was
                added), this is a no-op and returns ``False``.
        
                After removal the centroid is recomputed from the remaining vectors.
                If the window becomes empty, the centroid is reset to ``None``.
        
                Parameters
                ----------
                vector : np.ndarray
                    The exact array object to remove (identity comparison).
        
                Returns
                -------
                bool
                    ``True`` if the vector was found and removed; ``False`` if it
                    was not in the current window.
        
        Args:
            vector: See the function signature and surrounding type hints.
        
        Raises:
            None.
        """
        # Identity matching avoids removing a different ndarray with equal
        # values when duplicate vectors are present in the sliding window.
        found = False
        new_vectors: deque[np.ndarray] = deque(maxlen=self._config.window_size)
        for v in self._vectors:
            if not found and v is vector:
                found = True
            else:
                new_vectors.append(v)

        if not found:
            return False

        self._vectors = new_vectors
        if self._vectors:
            self._recalculate_centroid()
        else:
            self._current_centroid = None
        return True
