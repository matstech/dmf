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
tests/test_config_loader.py
---------------------------
Unit tests for threshold validation in dmf/utils/config_loader.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dmf.utils.config_loader import load_dmf_config


def _write_toml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "dmf_settings.toml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_dmf_config_accepts_ordered_memory_tiers(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[memory_tiers]
critical_max = 0.30
unstable_max = 0.60
healthy_min = 0.75
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.tiers.critical_max == 0.30
    assert cfg.tiers.unstable_max == 0.60
    assert cfg.tiers.healthy_min == 0.75


def test_load_dmf_config_rejects_critical_not_below_unstable(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[memory_tiers]
critical_max = 0.60
unstable_max = 0.60
healthy_min = 0.75
""".strip(),
    )

    with pytest.raises(
        ValueError,
        match="critical_max must be strictly lower than memory_tiers.unstable_max",
    ):
        load_dmf_config(path)


def test_load_dmf_config_rejects_unstable_above_healthy(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[memory_tiers]
critical_max = 0.30
unstable_max = 0.80
healthy_min = 0.75
""".strip(),
    )

    with pytest.raises(
        ValueError,
        match="unstable_max must be lower than or equal to memory_tiers.healthy_min",
    ):
        load_dmf_config(path)


def test_load_dmf_config_rejects_threshold_out_of_range(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[memory_tiers]
critical_max = -0.01
unstable_max = 0.60
healthy_min = 0.75
""".strip(),
    )

    with pytest.raises(
        ValueError,
        match="memory_tiers.critical_max must be within \\[0.0, 1.0\\]",
    ):
        load_dmf_config(path)


def test_load_dmf_config_rejects_unknown_ltm_storage_type(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "lancedb"
""".strip(),
    )

    with pytest.raises(ValueError, match=r"ltm.storage_type must be one of"):
        load_dmf_config(path)


def test_load_dmf_config_accepts_explicit_null_ltm_storage_type(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "null"
enabled = false
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.storage_type == "null"
    assert cfg.ltm.enabled is False


def test_load_dmf_config_parses_pruning_priority_section(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[pruning_priority]
rho_constraint = 0.25
rho_preference = 0.12
rho_current_state = 0.11
rho_correction = 0.18
rho_replacement = 0.09
superseded_past_penalty = 0.40
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.pruning_priority.rho_constraint == 0.25
    assert cfg.pruning_priority.rho_preference == 0.12
    assert cfg.pruning_priority.rho_current_state == 0.11
    assert cfg.pruning_priority.rho_correction == 0.18
    assert cfg.pruning_priority.rho_replacement == 0.09
    assert cfg.pruning_priority.superseded_past_penalty == 0.40


def test_load_dmf_config_parses_retrieval_candidate_generation_section(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[retrieval]
card_prefetch_k = 40
raw_prefetch_k = 12
symbolic_lookup_k = 9
final_recall_limit = 7
max_support_turns_per_card = 4
include_superseded_when_historical = false
include_neighbor_turns = true
enable_raw_semantic = false
enable_raw_lexical = true
enable_card_semantic = true
enable_card_symbolic = false
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.retrieval.card_prefetch_k == 40
    assert cfg.retrieval.raw_prefetch_k == 12
    assert cfg.retrieval.symbolic_lookup_k == 9
    assert cfg.retrieval.final_recall_limit == 7
    assert cfg.retrieval.max_support_turns_per_card == 4
    assert cfg.retrieval.include_superseded_when_historical is False
    assert cfg.retrieval.include_neighbor_turns is True
    assert cfg.retrieval.enable_raw_semantic is False
    assert cfg.retrieval.enable_raw_lexical is True
    assert cfg.retrieval.enable_card_semantic is True
    assert cfg.retrieval.enable_card_symbolic is False


def test_load_dmf_config_rejects_negative_retrieval_prefetch(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[retrieval]
raw_prefetch_k = -1
""".strip(),
    )

    with pytest.raises(ValueError, match="retrieval.raw_prefetch_k"):
        load_dmf_config(path)
