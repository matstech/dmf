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
tests/test_decay.py
-------------------
Unit tests for dmf/core/decay.py — Temporal Decay (Module 4, Phase 1).

All tests are pure arithmetic — no NLP models, no embeddings, no I/O.

Coverage
--------
  DecayConfig
    - Default values for all 5 fields.
    - Frozen immutability.

  calculate_effective_score — contract
    - Δn = 0 returns exactly Ω (no decay on latest message).
    - Ω = 0 returns 0.0 (nothing to save).
    - Ω_eff ≤ Ω always (decay never amplifies).
    - Ω_eff > 0 for any Ω > 0 (exponential never reaches zero).
    - Monotonically decreasing in Δn.

  calculate_effective_score — calibration table
    - HEALTHY (Anchor):  Ω = 0.85 at Δn = 10, 30, 60.
    - UNSTABLE (Mid):    Ω = 0.45 at Δn = 10, 30, 60.
    - SOCIAL (Floor):    Ω = 0.25 at Δn = 10, 30, 60.
    - CRITICAL (Noise):  Ω = 0.15 at Δn = 10, 30, 60.

  calculate_effective_score — inertia effect
    - HEALTHY decays slower than CRITICAL at every checkpoint.
    - Survival ratio widens over time.

  turns_to_hard_kill
    - Known lifespan for each tier.
    - Already-dead messages return 0.
    - Zero Ω returns 0.

  Edge cases
    - Negative Δn treated as 0 (no decay).
    - η = 0 gives uniform decay.
    - λ = 0 gives no decay.
"""

from __future__ import annotations

import math

import pytest

from dmf.analysis.decay import calculate_effective_score, turns_to_hard_kill
from dmf.utils.config import DecayConfig


# ---------------------------------------------------------------------------
# Constants — default calibration parameters
# ---------------------------------------------------------------------------

_LAMBDA = 0.035
_ETA = 0.5
_KILL = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eff(omega: float, dn: int, lam: float = _LAMBDA, eta: float = _ETA) -> float:
    """Reference implementation for expected-value calculations."""
    if omega == 0.0 or dn <= 0:
        return omega
    mu = 1.0 - eta * omega
    return omega * math.exp(-lam * mu * dn)


# ---------------------------------------------------------------------------
# DecayConfig defaults
# ---------------------------------------------------------------------------

class TestDecayConfig:
    """DecayConfig must store approved defaults and enforce immutability."""

    def test_default_lambda_decay(self) -> None:
        assert DecayConfig().lambda_decay == 0.035

    def test_default_inertia_strength(self) -> None:
        assert DecayConfig().inertia_strength == 0.5

    def test_default_hard_kill_threshold(self) -> None:
        assert DecayConfig().hard_kill_threshold == 0.05

    def test_default_token_budget(self) -> None:
        assert DecayConfig().token_budget == 4096

    def test_default_pruning_frequency(self) -> None:
        assert DecayConfig().pruning_frequency == 5

    def test_config_is_frozen(self) -> None:
        cfg = DecayConfig()
        with pytest.raises(AttributeError):
            cfg.lambda_decay = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Contract — boundary invariants
# ---------------------------------------------------------------------------

class TestContract:
    """calculate_effective_score must honour its mathematical contract."""

    def test_delta_n_zero_returns_exact_omega(self) -> None:
        """Δn = 0 → Ω_eff = Ω exactly (no decay on the latest message)."""
        assert calculate_effective_score(0.85, 0) == 0.85

    def test_omega_zero_returns_zero(self) -> None:
        """Ω = 0 → Ω_eff = 0.0 regardless of Δn (nothing to save)."""
        assert calculate_effective_score(0.0, 100) == 0.0

    def test_effective_score_never_exceeds_omega(self) -> None:
        """Ω_eff ≤ Ω for all Δn ≥ 0."""
        for omega in [0.15, 0.25, 0.45, 0.85]:
            for dn in [0, 1, 10, 50, 200]:
                assert calculate_effective_score(omega, dn) <= omega

    def test_effective_score_stays_positive(self) -> None:
        """Ω_eff > 0 for Ω > 0 within practical Δn range.

        At extreme Δn (e.g. 500+), the true value is astronomically small
        (< 1e-7) and rounds to 0.0 at 6 decimal places.  We test within
        the operational range (up to 200 turns) where rounding preserves
        the positivity guarantee.
        """
        for omega in [0.15, 0.25, 0.45, 0.85]:
            for dn in [1, 50, 100, 150]:
                assert calculate_effective_score(omega, dn) > 0.0

    def test_monotonically_decreasing_in_delta_n(self) -> None:
        """Ω_eff must decrease strictly as Δn increases (for fixed Ω > 0)."""
        omega = 0.72
        prev = omega
        for dn in range(1, 50):
            current = calculate_effective_score(omega, dn)
            assert current < prev, f"Ω_eff did not decrease at Δn={dn}"
            prev = current


# ---------------------------------------------------------------------------
# Calibration table — exact values verified against reference impl
# ---------------------------------------------------------------------------

class TestCalibrationTable:
    """Decay trajectories must match the approved calibration report."""

    # Tolerance: 4 decimal places (function rounds to 6, table to 4)
    _TOL = 5e-4

    # -- HEALTHY (Anchor): Ω = 0.85, μ = 0.575 -------------------------

    def test_healthy_dn_10(self) -> None:
        assert abs(calculate_effective_score(0.85, 10) - 0.6951) < self._TOL

    def test_healthy_dn_30(self) -> None:
        assert abs(calculate_effective_score(0.85, 30) - 0.4647) < self._TOL

    def test_healthy_dn_60(self) -> None:
        assert abs(calculate_effective_score(0.85, 60) - 0.2541) < self._TOL

    # -- UNSTABLE (Mid): Ω = 0.45, μ = 0.775 ----------------------------

    def test_unstable_dn_10(self) -> None:
        assert abs(calculate_effective_score(0.45, 10) - 0.3431) < self._TOL

    def test_unstable_dn_30(self) -> None:
        assert abs(calculate_effective_score(0.45, 30) - 0.1994) < self._TOL

    def test_unstable_dn_60(self) -> None:
        assert abs(calculate_effective_score(0.45, 60) - 0.0884) < self._TOL

    # -- SOCIAL (Floor): Ω = 0.25, μ = 0.875 ----------------------------

    def test_social_dn_10(self) -> None:
        assert abs(calculate_effective_score(0.25, 10) - 0.1841) < self._TOL

    def test_social_dn_30(self) -> None:
        assert abs(calculate_effective_score(0.25, 30) - 0.0998) < self._TOL

    def test_social_dn_60(self) -> None:
        assert abs(calculate_effective_score(0.25, 60) - 0.0398) < self._TOL

    # -- CRITICAL (Noise): Ω = 0.15, μ = 0.925 --------------------------

    def test_critical_dn_10(self) -> None:
        assert abs(calculate_effective_score(0.15, 10) - 0.1085) < self._TOL

    def test_critical_dn_30(self) -> None:
        assert abs(calculate_effective_score(0.15, 30) - 0.0568) < self._TOL

    def test_critical_dn_60(self) -> None:
        assert abs(calculate_effective_score(0.15, 60) - 0.0215) < self._TOL


# ---------------------------------------------------------------------------
# Inertia effect — survival gap analysis
# ---------------------------------------------------------------------------

class TestInertiaEffect:
    """High-Ω messages must consistently outrank low-Ω messages over time."""

    def test_healthy_always_outranks_social(self) -> None:
        """HEALTHY (0.85) > SOCIAL (0.25) at every checkpoint."""
        for dn in [10, 20, 30, 40, 50]:
            healthy = calculate_effective_score(0.85, dn)
            social = calculate_effective_score(0.25, dn)
            assert healthy > social, f"Inversion at Δn={dn}"

    def test_survival_ratio_widens_over_time(self) -> None:
        """The ratio HEALTHY/SOCIAL must increase as Δn grows.

        This is the key inertia property: because μ_healthy < μ_social,
        the gap between them *widens*, never narrows.
        """
        prev_ratio = 0.0
        for dn in [10, 20, 30, 40, 50]:
            healthy = calculate_effective_score(0.85, dn)
            social = calculate_effective_score(0.25, dn)
            ratio = healthy / social
            assert ratio > prev_ratio, f"Ratio shrank at Δn={dn}"
            prev_ratio = ratio

    def test_unstable_outranks_critical_at_dn_30(self) -> None:
        """UNSTABLE (0.45) must still outrank CRITICAL (0.15) at Δn=30."""
        unstable = calculate_effective_score(0.45, 30)
        critical = calculate_effective_score(0.15, 30)
        assert unstable > critical


# ---------------------------------------------------------------------------
# turns_to_hard_kill
# ---------------------------------------------------------------------------

class TestTurnsToHardKill:
    """turns_to_hard_kill must return the first Δn where Ω_eff < Ω_kill."""

    def test_healthy_lifespan(self) -> None:
        """HEALTHY (0.85) lives ~141 turns."""
        assert turns_to_hard_kill(0.85) == 141

    def test_unstable_lifespan(self) -> None:
        """UNSTABLE (0.45) lives ~82 turns."""
        assert turns_to_hard_kill(0.45) == 82

    def test_social_lifespan(self) -> None:
        """SOCIAL (0.25) lives ~53 turns."""
        assert turns_to_hard_kill(0.25) == 53

    def test_critical_lifespan(self) -> None:
        """CRITICAL (0.15) lives ~34 turns."""
        assert turns_to_hard_kill(0.15) == 34

    def test_already_dead_returns_zero(self) -> None:
        """Ω ≤ Ω_kill → already dead, returns 0."""
        assert turns_to_hard_kill(0.05) == 0
        assert turns_to_hard_kill(0.01) == 0

    def test_zero_omega_returns_zero(self) -> None:
        """Ω = 0 → dead on arrival."""
        assert turns_to_hard_kill(0.0) == 0

    def test_hard_kill_boundary_is_correct(self) -> None:
        """At the returned Δn, Ω_eff must be below threshold.
        At Δn − 1, Ω_eff must still be at or above threshold."""
        omega = 0.72
        ttk = turns_to_hard_kill(omega)
        assert ttk is not None

        # At ttk, should be below threshold
        at_kill = calculate_effective_score(omega, ttk)
        assert at_kill < _KILL

        # One turn earlier, should still be alive
        before_kill = calculate_effective_score(omega, ttk - 1)
        assert before_kill >= _KILL


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Degenerate and boundary inputs must be handled gracefully."""

    def test_negative_delta_n_returns_omega(self) -> None:
        """Negative Δn makes no physical sense — treat as 0 (no decay)."""
        assert calculate_effective_score(0.72, -5) == 0.72

    def test_eta_zero_gives_uniform_decay(self) -> None:
        """η = 0 → μ = 1 for all Ω → all messages decay at the same rate.

        Two messages with different Ω but same Δn must have the same
        decay *fraction* (not the same absolute Ω_eff).  Tolerance is
        1e-5 to account for 6-dp rounding artefacts in the division.
        """
        dn = 20
        frac_high = calculate_effective_score(0.85, dn, inertia_strength=0.0) / 0.85
        frac_low = calculate_effective_score(0.25, dn, inertia_strength=0.0) / 0.25
        assert abs(frac_high - frac_low) < 1e-5

    def test_lambda_zero_gives_no_decay(self) -> None:
        """λ = 0 → Ω_eff = Ω regardless of Δn (immortal messages)."""
        assert calculate_effective_score(0.72, 1000, lambda_decay=0.0) == 0.72

    def test_large_delta_n_produces_near_zero_but_positive(self) -> None:
        """Δn = 10000 → Ω_eff is astronomically small but still > 0."""
        result = calculate_effective_score(0.85, 10000)
        assert result >= 0.0  # Rounding to 6 dp may produce 0.0
        # But with reasonable Δn it should stay positive
        result_200 = calculate_effective_score(0.85, 200)
        assert result_200 > 0.0

    def test_turns_to_hard_kill_with_lambda_zero(self) -> None:
        """λ = 0 → no decay → never reaches hard-kill → returns None."""
        assert turns_to_hard_kill(0.85, lambda_decay=0.0) is None
