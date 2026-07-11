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

from __future__ import annotations

import pytest

from dmf.memory.ltm_hooks.vector_types import (
    cosine_distance_to_similarity,
    cosine_similarity_to_distance,
    distance_threshold_to_min_similarity,
    validate_vector_dimension,
)


@pytest.mark.parametrize("value", [1.0, 0.3, 0.0, -0.5])
def test_cosine_score_conversions_use_one_minus_value(value: float) -> None:
    assert cosine_distance_to_similarity(value) == pytest.approx(1.0 - value)
    assert cosine_similarity_to_distance(value) == pytest.approx(1.0 - value)
    assert distance_threshold_to_min_similarity(value) == pytest.approx(1.0 - value)


def test_cosine_score_conversions_do_not_clamp_negative_scores() -> None:
    assert cosine_distance_to_similarity(-0.5) == pytest.approx(1.5)
    assert cosine_similarity_to_distance(-0.5) == pytest.approx(1.5)
    assert distance_threshold_to_min_similarity(-0.5) == pytest.approx(1.5)


def test_validate_vector_dimension_accepts_matching_size() -> None:
    validate_vector_dimension([0.1, 0.2, 0.3], 3, field="query_vector")


def test_validate_vector_dimension_rejects_mismatch_with_context() -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_vector_dimension([0.1, 0.2], 3, field="query_vector")

    message = str(exc_info.value)
    assert "query_vector" in message
    assert "expected 3" in message
    assert "observed 2" in message
