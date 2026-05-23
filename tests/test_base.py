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

"""
tests/test_base.py
------------------
Bootstrap validation tests for dmf/runtime/base.py.

These tests confirm that:
  - AuditState can be instantiated with required fields only.
  - All optional fields default correctly.
  - to_dict() serialises every field without data loss.
  - The _AuditTimer context-manager measures latency and builds AuditState.
  - Latency is non-negative and populated.
"""

import time

import pytest

import dmf as dmf_pkg
from dmf.runtime.base import AuditState


# ---------------------------------------------------------------------------
# AuditState – construction & defaults
# ---------------------------------------------------------------------------

def test_audit_state_minimal_construction():
    """AuditState must be constructible with only module_name and operation."""
    state = AuditState(module_name="nlp_engine", operation="extract_features")

    assert state.module_name == "nlp_engine"
    assert state.operation == "extract_features"
    assert state.latency_ms == 0.0
    assert state.inputs == {}
    assert state.intermediate_weights == {}
    assert state.logical_decisions == {}
    assert state.metadata == {}
    assert state.created_at > 0


def test_audit_state_full_construction():
    """AuditState must store every explicit field without mutation."""
    state = AuditState(
        module_name="scoring_engine",
        operation="calculate_survival_score",
        latency_ms=3.14,
        inputs={"text": "hello world"},
        intermediate_weights={"information_density": 0.5, "sentiment_abs": 0.2},
        logical_decisions={"social_floor_triggered": False},
        metadata={"model": "en_core_web_sm"},
    )

    assert state.latency_ms == 3.14
    assert state.inputs["text"] == "hello world"
    assert state.intermediate_weights["information_density"] == 0.5
    assert state.logical_decisions["social_floor_triggered"] is False
    assert state.metadata["model"] == "en_core_web_sm"


# ---------------------------------------------------------------------------
# AuditState.to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_contains_all_keys():
    """to_dict() must expose every AuditState field for JSON logging."""
    state = AuditState(
        module_name="geometry_engine",
        operation="compute_cosine_divergence",
        intermediate_weights={"cosine_distance": 0.33},
    )
    result = state.to_dict()

    expected_keys = {
        "module_name", "operation", "latency_ms",
        "inputs", "intermediate_weights",
        "logical_decisions", "metadata", "created_at",
    }
    assert expected_keys == set(result.keys())
    assert result["intermediate_weights"]["cosine_distance"] == 0.33


def test_to_dict_values_are_consistent_with_state():
    """to_dict() must not silently mutate or lose field values."""
    state = AuditState(
        module_name="pruning_engine",
        operation="evict_interaction",
        latency_ms=12.5,
        logical_decisions={"landmark_protected": True},
    )
    d = state.to_dict()

    assert d["module_name"] == state.module_name
    assert d["latency_ms"] == state.latency_ms
    assert d["logical_decisions"] == state.logical_decisions


# ---------------------------------------------------------------------------
# AuditState.start_timer() / _AuditTimer
# ---------------------------------------------------------------------------

def test_start_timer_produces_audit_state():
    """start_timer context-manager must set .state after block exit."""
    with AuditState.start_timer("nlp_engine", "tokenize") as timer:
        time.sleep(0.001)  # simulate work

    assert timer.state is not None
    assert isinstance(timer.state, AuditState)


def test_start_timer_latency_is_positive():
    """Recorded latency must be > 0 when the block performs any work."""
    with AuditState.start_timer("scoring_engine", "weight_calculation") as timer:
        _ = sum(range(10_000))  # deterministic CPU work

    assert timer.state is not None
    assert timer.state.latency_ms > 0


def test_start_timer_captures_intermediate_weights():
    """Intermediate weights set inside the block must appear in the state."""
    with AuditState.start_timer("nlp_engine", "density_extraction") as timer:
        timer.intermediate_weights = {"density": 0.75, "entity_count": 3}
        timer.logical_decisions = {"gpu_used": False}

    assert timer.state is not None
    assert timer.state.intermediate_weights["density"] == 0.75
    assert timer.state.logical_decisions["gpu_used"] is False


def test_start_timer_module_and_operation_fields():
    """module_name and operation must propagate correctly into the state."""
    with AuditState.start_timer("storage_layer", "archive_interaction") as timer:
        pass  # no-op block

    assert timer.state is not None
    assert timer.state.module_name == "storage_layer"
    assert timer.state.operation == "archive_interaction"


def test_audit_state_is_not_reexported_from_dmf_package():
    """AuditState stays available only from `dmf.runtime.base`, not from `dmf`."""
    assert "AuditState" not in dmf_pkg.__all__
