# Copyright (c) 2026-present matstech
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT

"""Vector score and shape helpers shared by LTM backends."""

from __future__ import annotations

from collections.abc import Sequence


def cosine_distance_to_similarity(distance: float) -> float:
    """Convert cosine distance to similarity without clamping."""
    return 1.0 - distance


def cosine_similarity_to_distance(score: float) -> float:
    """Convert cosine similarity to distance without clamping."""
    return 1.0 - score


def distance_threshold_to_min_similarity(threshold: float) -> float:
    """Convert a maximum distance threshold into a minimum similarity."""
    return 1.0 - threshold


def validate_vector_dimension(
    vector: Sequence[float],
    expected: int,
    *,
    field: str,
) -> None:
    """Raise when a vector does not match the configured dimensionality."""
    observed = len(vector)
    if observed != expected:
        raise ValueError(
            f"{field} vector dimension mismatch: expected {expected}, observed {observed}"
        )
