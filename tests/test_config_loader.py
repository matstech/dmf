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

from dmf.utils.constants import LTM_BACKEND_QDRANT, SUPPORTED_LTM_BACKENDS
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


def test_ltm_chroma_connection_defaults_preserve_embedded_mode(
    tmp_path: Path,
) -> None:
    cfg = load_dmf_config(_write_toml(tmp_path, "[ltm]"))

    assert cfg.ltm.qdrant_mode == "memory"
    assert cfg.ltm.qdrant_host == "localhost"
    assert cfg.ltm.qdrant_port == 6333
    assert cfg.ltm.qdrant_ssl is False
    assert cfg.ltm.qdrant_api_key_env == ""
    assert cfg.ltm.qdrant_timeout == 5
    assert cfg.ltm.chroma_mode == "embedded"
    assert cfg.ltm.chroma_host == "localhost"
    assert cfg.ltm.chroma_port == 8000
    assert cfg.ltm.chroma_ssl is False
    assert cfg.ltm.chroma_tenant == "default_tenant"
    assert cfg.ltm.chroma_database == "default_database"
    assert cfg.ltm.chroma_auth_token_env == ""


def test_load_dmf_config_parses_chroma_server_settings(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "chroma"
enabled = true
chroma_mode = "server"
chroma_host = "chroma.internal"
chroma_port = 8443
chroma_ssl = true
chroma_tenant = "tenant-a"
chroma_database = "database-a"
chroma_auth_token_env = "DMF_CHROMA_TOKEN"
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.chroma_mode == "server"
    assert cfg.ltm.chroma_host == "chroma.internal"
    assert cfg.ltm.chroma_port == 8443
    assert cfg.ltm.chroma_ssl is True
    assert cfg.ltm.chroma_tenant == "tenant-a"
    assert cfg.ltm.chroma_database == "database-a"
    assert cfg.ltm.chroma_auth_token_env == "DMF_CHROMA_TOKEN"


def test_supported_ltm_backends_include_qdrant() -> None:
    assert LTM_BACKEND_QDRANT in SUPPORTED_LTM_BACKENDS


def test_load_dmf_config_parses_explicit_qdrant_storage_type(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "qdrant"
qdrant_mode = "memory"
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.storage_type == "qdrant"
    assert cfg.ltm.qdrant_mode == "memory"


def test_load_dmf_config_parses_qdrant_server_settings(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "qdrant"
qdrant_mode = "server"
qdrant_host = "qdrant.internal"
qdrant_port = 7443
qdrant_ssl = true
qdrant_api_key_env = "DMF_QDRANT_API_KEY"
qdrant_timeout = 12
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.qdrant_mode == "server"
    assert cfg.ltm.qdrant_host == "qdrant.internal"
    assert cfg.ltm.qdrant_port == 7443
    assert cfg.ltm.qdrant_ssl is True
    assert cfg.ltm.qdrant_api_key_env == "DMF_QDRANT_API_KEY"
    assert cfg.ltm.qdrant_timeout == 12


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("qdrant_host", '"   "', "ltm.qdrant_host"),
        ("qdrant_port", "0", "ltm.qdrant_port"),
        ("qdrant_port", "65536", "ltm.qdrant_port"),
        ("qdrant_timeout", "0", "ltm.qdrant_timeout"),
        ("qdrant_api_key_env", '"   "', "ltm.qdrant_api_key_env"),
    ],
)
def test_load_dmf_config_rejects_invalid_qdrant_server_settings(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                'storage_type = "qdrant"',
                'qdrant_mode = "server"',
                f"{field} = {value}",
            ]
        ),
    )

    with pytest.raises(ValueError, match=message):
        load_dmf_config(path)


def test_load_dmf_config_rejects_unknown_chroma_mode_when_chroma_active(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "chroma"
enabled = true
chroma_mode = "cluster"
""".strip(),
    )

    with pytest.raises(ValueError, match=r"ltm.chroma_mode must be one of"):
        load_dmf_config(path)


@pytest.mark.parametrize(
    ("storage_type", "enabled"),
    [("file", True), ("null", True), ("qdrant", True), ("chroma", False)],
)
def test_load_dmf_config_ignores_unknown_chroma_mode_when_chroma_inactive(
    tmp_path: Path,
    storage_type: str,
    enabled: bool,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                f'storage_type = "{storage_type}"',
                f"enabled = {str(enabled).lower()}",
                'chroma_mode = "cluster"',
            ]
        ),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.chroma_mode == "cluster"


def test_load_dmf_config_rejects_unknown_qdrant_mode_when_qdrant_active(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "qdrant"
enabled = true
qdrant_mode = "disk"
""".strip(),
    )

    with pytest.raises(ValueError, match=r"ltm.qdrant_mode must be one of"):
        load_dmf_config(path)


@pytest.mark.parametrize(
    ("storage_type", "enabled"),
    [("file", True), ("null", True), ("chroma", True), ("qdrant", False)],
)
def test_load_dmf_config_ignores_unknown_qdrant_mode_when_qdrant_inactive(
    tmp_path: Path,
    storage_type: str,
    enabled: bool,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                f'storage_type = "{storage_type}"',
                f"enabled = {str(enabled).lower()}",
                'qdrant_mode = "disk"',
            ]
        ),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.qdrant_mode == "disk"


def test_load_dmf_config_ignores_invalid_chroma_fields_when_qdrant_active(
    tmp_path: Path,
) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "qdrant"
enabled = true
qdrant_mode = "memory"
chroma_mode = "cluster"
chroma_host = ""
chroma_port = 0
chroma_tenant = ""
chroma_database = ""
chroma_auth_token_env = "   "
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.storage_type == "qdrant"
    assert cfg.ltm.qdrant_mode == "memory"


@pytest.mark.parametrize(
    ("storage_type", "enabled"),
    [("file", True), ("null", True), ("chroma", True), ("qdrant", False)],
)
def test_inactive_qdrant_server_does_not_validate_connection_fields(
    tmp_path: Path,
    storage_type: str,
    enabled: bool,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                f'storage_type = "{storage_type}"',
                f"enabled = {str(enabled).lower()}",
                'qdrant_mode = "server"',
                'qdrant_host = ""',
                "qdrant_port = 0",
                "qdrant_timeout = 0",
                'qdrant_api_key_env = "   "',
            ]
        ),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.qdrant_mode == "server"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chroma_host", '"   "', "ltm.chroma_host"),
        ("chroma_tenant", '""', "ltm.chroma_tenant"),
        ("chroma_database", '"  "', "ltm.chroma_database"),
    ],
)
def test_load_dmf_config_rejects_empty_active_server_identifiers(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                'storage_type = "chroma"',
                'chroma_mode = "server"',
                f"{field} = {value}",
            ]
        ),
    )

    with pytest.raises(ValueError, match=message):
        load_dmf_config(path)


@pytest.mark.parametrize("port", [0, -1, 65536])
def test_load_dmf_config_rejects_invalid_active_server_port(
    tmp_path: Path,
    port: int,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                'storage_type = "chroma"',
                'chroma_mode = "server"',
                f"chroma_port = {port}",
            ]
        ),
    )

    with pytest.raises(ValueError, match="ltm.chroma_port"):
        load_dmf_config(path)


@pytest.mark.parametrize(
    ("storage_type", "enabled"),
    [("file", True), ("null", True), ("chroma", False)],
)
def test_inactive_chroma_server_does_not_validate_connection_fields(
    tmp_path: Path,
    storage_type: str,
    enabled: bool,
) -> None:
    path = _write_toml(
        tmp_path,
        "\n".join(
            [
                "[ltm]",
                f'storage_type = "{storage_type}"',
                f"enabled = {str(enabled).lower()}",
                'chroma_mode = "server"',
                'chroma_host = ""',
                "chroma_port = 0",
                'chroma_tenant = ""',
                'chroma_database = ""',
            ]
        ),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.chroma_mode == "server"


def test_load_dmf_config_rejects_whitespace_auth_env_name(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
storage_type = "chroma"
chroma_auth_token_env = "   "
""".strip(),
    )

    with pytest.raises(ValueError, match="ltm.chroma_auth_token_env"):
        load_dmf_config(path)


def test_load_dmf_config_rejects_negative_ltm_recall_limit(tmp_path: Path) -> None:
    path = _write_toml(
        tmp_path,
        """
[ltm]
recall_limit = -1
""".strip(),
    )

    with pytest.raises(ValueError, match="ltm.recall_limit"):
        load_dmf_config(path)


@pytest.mark.parametrize("threshold", [-0.1, 2.1])
def test_load_dmf_config_rejects_ltm_distance_threshold_out_of_range(
    tmp_path: Path,
    threshold: float,
) -> None:
    path = _write_toml(
        tmp_path,
        f"""
[ltm]
distance_threshold = {threshold}
""".strip(),
    )

    with pytest.raises(ValueError, match=r"ltm.distance_threshold"):
        load_dmf_config(path)


@pytest.mark.parametrize("threshold", [0.0, 2.0])
def test_load_dmf_config_accepts_ltm_distance_threshold_boundaries(
    tmp_path: Path,
    threshold: float,
) -> None:
    path = _write_toml(
        tmp_path,
        f"""
[ltm]
distance_threshold = {threshold}
""".strip(),
    )

    cfg = load_dmf_config(path)

    assert cfg.ltm.distance_threshold == threshold


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
