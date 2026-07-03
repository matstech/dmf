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
tests/test_verify_config.py
---------------------------
Pytest suite for TOML config loading and component injection.

Migrated from scripts/verify_config.py into the standard test suite.
"""

from __future__ import annotations

import pytest

from dmf.runtime.pipeline import InteractionPipeline
from dmf.analysis.geometry import InteractionMatrix
from dmf.analysis.scoring_engine import ScoringEngine
from dmf.memory.temporal_memory import TemporalMemory
from dmf.utils.config_loader import DMFConfig, load_dmf_config


@pytest.fixture(scope="module")
def cfg() -> DMFConfig:
    return load_dmf_config()


class TestConfigParsing:
    """TOML is parsed and all sections are populated."""

    def test_load_succeeds(self, cfg: DMFConfig) -> None:
        assert cfg is not None

    def test_nlp_section(self, cfg: DMFConfig) -> None:
        assert isinstance(cfg.nlp.spacy_model, str)
        assert cfg.nlp.vector_dim > 0

    def test_ltm_chroma_connection_section(self, cfg: DMFConfig) -> None:
        assert cfg.ltm.chroma_mode == "embedded"
        assert cfg.ltm.chroma_host == "localhost"
        assert cfg.ltm.chroma_port == 8000
        assert cfg.ltm.chroma_ssl is False
        assert cfg.ltm.chroma_tenant == "default_tenant"
        assert cfg.ltm.chroma_database == "default_database"
        assert cfg.ltm.chroma_auth_token_env == ""


class TestComponentInjection:
    """Factory classmethods produce valid components from DMFConfig."""

    def test_scoring_engine(self, cfg: DMFConfig) -> None:
        engine = ScoringEngine.from_dmf_config(cfg)
        assert engine is not None
        assert engine._config.critical_threshold == cfg.tiers.critical_max
        assert engine._config.healthy_threshold == cfg.tiers.healthy_min
        assert engine._config.lambda_operational == cfg.scoring.lambda_operational
        assert engine._config.eta_constraint == cfg.scoring.eta_constraint
        assert engine._config.eta_preference == cfg.scoring.eta_preference
        assert engine._config.eta_current_state == cfg.scoring.eta_current_state
        assert engine._config.eta_correction == cfg.scoring.eta_correction
        assert engine._config.eta_replacement == cfg.scoring.eta_replacement
        assert engine._config.eta_past_state == cfg.scoring.eta_past_state
        assert engine._config.user_correction_boost == cfg.scoring.user_correction_boost
        assert engine._config.preference_update_boost == cfg.scoring.preference_update_boost
        assert engine._config.constraint_boost == cfg.scoring.constraint_boost
        assert engine._config.corrected_by_user_penalty == cfg.scoring.corrected_by_user_penalty

    def test_interaction_matrix(self, cfg: DMFConfig) -> None:
        matrix = InteractionMatrix.from_dmf_config(cfg)
        assert matrix.window_size == cfg.capacity.window_size

    def test_interaction_pipeline(self, cfg: DMFConfig) -> None:
        pipeline = InteractionPipeline.from_dmf_config(
            cfg,
            analyze_system_prompt=False,
        )
        assert pipeline._nlp_config.spacy_model == cfg.nlp.spacy_model
        assert pipeline._nlp_config.analyze_system_prompt is False
        assert pipeline._vector_config.vector_dim == cfg.nlp.vector_dim
        assert pipeline._vector_config.window_size == cfg.capacity.window_size

    def test_temporal_memory(self, cfg: DMFConfig) -> None:
        tm = TemporalMemory.from_dmf_config(cfg)
        assert tm.config.token_budget == cfg.capacity.token_budget
        assert tm.config.lambda_decay == cfg.decay.lambda_base
        assert tm.config.critical_threshold == cfg.tiers.critical_max
        assert tm.config.healthy_threshold == cfg.tiers.healthy_min
        assert tm._pruning_priority_config.rho_constraint == cfg.pruning_priority.rho_constraint
        assert tm._pruning_priority_config.rho_preference == cfg.pruning_priority.rho_preference
        assert (
            tm._pruning_priority_config.rho_current_state
            == cfg.pruning_priority.rho_current_state
        )
        assert tm._pruning_priority_config.rho_correction == cfg.pruning_priority.rho_correction
        assert tm._pruning_priority_config.rho_replacement == cfg.pruning_priority.rho_replacement
        assert (
            tm._pruning_priority_config.superseded_past_penalty
            == cfg.pruning_priority.superseded_past_penalty
        )
