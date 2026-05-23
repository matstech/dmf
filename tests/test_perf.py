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
tests/test_perf.py
------------------
Unit tests for dmf/utils/perf.py — ExecutionLatencyTimer context manager.

Coverage:
  - elapsed_ms is positive after a non-trivial block (CPU work).
  - elapsed_ms reflects real duration for a timed sleep (accuracy check).
  - elapsed_ms is accessible only after the 'with' block exits.
  - Accessing elapsed_ms on an unused instance raises RuntimeError.
  - RuntimeError message is descriptive enough to guide the caller.
  - Timer reuse: each 'with' block produces an independent measurement.
  - Timer reuse: re-entering resets the previous elapsed_ms to None.
  - __enter__ returns the timer instance itself (supports 'with X as t').
"""

import time

import pytest

from dmf.utils.perf import ExecutionLatencyTimer


# ---------------------------------------------------------------------------
# Basic measurement
# ---------------------------------------------------------------------------

def test_elapsed_ms_is_positive_after_cpu_work():
    """elapsed_ms must be > 0 after a block that performs real CPU work."""
    timer = ExecutionLatencyTimer()
    with timer:
        _ = sum(range(100_000))  # deterministic CPU work
    assert timer.elapsed_ms > 0


def test_elapsed_ms_reflects_sleep_duration():
    """elapsed_ms must be >= 9.0 after a 10 ms sleep (accounts for OS jitter)."""
    timer = ExecutionLatencyTimer()
    with timer:
        time.sleep(0.01)  # 10 ms nominal
    # 9 ms lower bound: conservatively accounts for OS scheduling variance
    # while still proving the timer measured a real 10 ms sleep.
    assert timer.elapsed_ms >= 9.0


def test_elapsed_ms_returns_a_float():
    """elapsed_ms must be a float, not an int or other numeric type."""
    timer = ExecutionLatencyTimer()
    with timer:
        pass
    assert isinstance(timer.elapsed_ms, float)


# ---------------------------------------------------------------------------
# Guard: accessing elapsed_ms before measurement
# ---------------------------------------------------------------------------

def test_elapsed_ms_raises_before_any_with_block():
    """Accessing elapsed_ms on a fresh, never-entered instance must raise RuntimeError."""
    timer = ExecutionLatencyTimer()
    with pytest.raises(RuntimeError):
        _ = timer.elapsed_ms


def test_elapsed_ms_error_message_is_descriptive():
    """The RuntimeError message must mention the 'with' block requirement."""
    timer = ExecutionLatencyTimer()
    with pytest.raises(RuntimeError, match="with"):
        _ = timer.elapsed_ms


# ---------------------------------------------------------------------------
# Reusability
# ---------------------------------------------------------------------------

def test_reuse_produces_independent_measurements():
    """Two sequential 'with' blocks must yield independently measured durations."""
    timer = ExecutionLatencyTimer()

    with timer:
        time.sleep(0.01)   # ~10 ms
    first_elapsed_ms = timer.elapsed_ms

    with timer:
        time.sleep(0.02)   # ~20 ms
    second_elapsed_ms = timer.elapsed_ms

    # Both must be positive and independently measured.
    assert first_elapsed_ms > 0
    assert second_elapsed_ms > 0
    # Second sleep is longer; its duration must be strictly greater.
    assert second_elapsed_ms > first_elapsed_ms


def test_reuse_resets_elapsed_ms_to_unavailable_on_entry():
    """Re-entering a used timer must make elapsed_ms unavailable until exit."""
    timer = ExecutionLatencyTimer()

    # First use: elapsed_ms becomes available.
    with timer:
        pass
    assert timer.elapsed_ms >= 0  # available after first exit

    # Manually enter without exiting to simulate mid-block access.
    timer.__enter__()
    with pytest.raises(RuntimeError):
        _ = timer.elapsed_ms  # must be unavailable again
    timer.__exit__(None, None, None)  # clean up


# ---------------------------------------------------------------------------
# Context manager protocol
# ---------------------------------------------------------------------------

def test_enter_returns_the_timer_instance():
    """__enter__ must return the timer itself to support 'with X as t' syntax."""
    timer = ExecutionLatencyTimer()
    with timer as context_target:
        assert context_target is timer
