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
tests/test_ltm_persistence.py
------------------------------
Module 6 — Long-Term Memory persistence tests.

Coverage
--------
  MemoryEntry.to_dict
    - All eight keys are present in the returned dict.
    - interaction_id is an int.
    - text is the original string.
    - token_count matches the entry.
    - timestamp matches the entry.
    - omega matches entry.omega (survival_score from report).
    - status is a string (SurvivalStatus.value) or None.
    - vector is a plain Python list (not np.ndarray).
    - vector values match entry.vector.tolist().
    - report key contains the full AnalysisReport dict.
    - Unscored entry (survival_score=None) serialises omega as 0.0.

  FileLTMHook — construction
    - Parent directory is created automatically if it does not exist.
    - Deeply nested parent directories are created.
    - The archive file does not exist until the first archive() call.
    - .path property returns the resolved Path.

  FileLTMHook — archive()
    - Creates the file on first call.
    - Writes exactly one line per archive() call.
    - Each line is valid JSON (json.loads succeeds).
    - Appends; does not overwrite existing content.
    - text field matches entry.text.
    - omega field matches entry.omega.
    - vector field is a list of floats matching entry.vector.tolist().
    - interaction_id field matches entry.interaction_id.
    - token_count field matches entry.token_count.
    - status field is a string matching entry.status.value.
    - report key is present and is a dict.
    - Satisfies the LTMHook Protocol (isinstance check).
    - Two archive() calls produce two lines in insertion order.
    - ensure_ascii=False: non-ASCII characters are preserved verbatim.

  LTMSettings
    - Default values match the TOML defaults.
    - Frozen (immutable) — assignment raises FrozenInstanceError.

  DMFConfig.ltm
    - DMFConfig() (no-arg) carries a default LTMSettings.
    - load_dmf_config() parses the [ltm] section from the TOML file.
    - Parsed storage_path matches the TOML value.
    - Parsed enabled matches the TOML value.

  TemporalMemory.from_dmf_config — LTM hook resolution
    - When ltm.enabled=True and storage_type="file", from_dmf_config
      creates a FileLTMHook pointing to ltm.storage_path.
    - Explicit ltm_hook= injection overrides TOML config.
    - When ltm.enabled=False, NullLTMHook is used (no file created).

  Integration — eviction → JSONL archive
    - A CRITICAL entry evicted by prune_to_budget appears in the archive.
    - The archived text matches the original message text.
    - The archived omega matches the original entry.omega.
    - The archived vector matches the original entry.vector (round-trip).
    - A CRITICAL entry evicted by periodic_cleanup appears in the archive.
    - Multiple evictions from one prune_to_budget call each produce one line.
    - Archive file is append-only: pre-existing lines are not overwritten.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest

import dmf.memory.ltm_hooks as ltm_hooks
from dmf.memory.ltm_hooks import FileLTMHook
from dmf.models.analysis import (
    AnalysisReport,
    InteractionProvenance,
    InteractionSignals,
    MemoryLineage,
)
from dmf.memory.temporal_memory import TemporalMemory
from dmf.models.ltm_hook import LTMHook, NullLTMHook
from dmf.models.memory import MemoryEntry
from dmf.models.status import SurvivalStatus, classify_survival_status
from dmf.utils.config import DecayConfig, VectorConfig
from dmf.utils.config_loader import (
    CapacitySettings,
    DMFConfig,
    LTMSettings,
    MemoryTiersSettings,
    NLPSettings,
    ScoringWeightsSettings,
    TemporalDecaySettings,
    load_dmf_config,
)


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_report(omega: float = 0.15) -> AnalysisReport:
    """Build a minimal scored AnalysisReport."""
    report = AnalysisReport(
        info_density=0.4,
        sentiment_abs=0.1,
        entity_count=1,
        is_system_prompt=False,
        latency_ms=1.0,
        semantic_divergence=0.05,
        survival_score=omega,
        status=classify_survival_status(
            omega=omega,
            critical_threshold=0.3,
            healthy_threshold=0.6,
        ),
        provenance=InteractionProvenance(
            role="assistant",
            source_turn=2,
            derived_from_model=True,
        ),
    )
    return report


def _unit_vector(dim: int, seed: int = 0) -> np.ndarray:
    """Return a deterministic L2-normalised vector of given dimension."""
    rng = np.random.default_rng(seed)
    v = rng.random(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_entry(
    text: str = "hello world",
    omega: float = 0.15,
    seed: int = 0,
    dim: int = 8,
    interaction_id: int = 0,
) -> MemoryEntry:
    """Build a MemoryEntry without going through TemporalMemory."""
    return MemoryEntry(
        interaction_id=interaction_id,
        text=text,
        report=_make_report(omega=omega),
        vector=_unit_vector(dim, seed),
        token_count=len(text.split()),
        timestamp=time.time(),
    )


_VECTOR_CFG = VectorConfig(window_size=5, vector_dim=8)


def _budget_tm(
    budget: int,
    hook: LTMHook,
    pruning_frequency: int = 999_999,
) -> TemporalMemory:
    """Return a TemporalMemory configured for deterministic budget-pressure tests."""
    cfg = DecayConfig(token_budget=budget, pruning_frequency=pruning_frequency)
    return TemporalMemory(
        decay_config=cfg,
        vector_config=_VECTOR_CFG,
        ltm_hook=hook,
    )


# ===========================================================================
# MemoryEntry.to_dict
# ===========================================================================

class TestMemoryEntryToDict:
    """MemoryEntry.to_dict() — serialisation correctness."""

    def setup_method(self) -> None:
        self.entry = _make_entry(
            text="Alice booked three tickets to Paris.",
            omega=0.87,
            seed=1,
            dim=8,
            interaction_id=7,
        )
        self.entry.report.signals = InteractionSignals(
            is_preference=True,
            is_current_state=True,
            cue_phrases=["my favorite"],
        )
        self.entry.report.topic_identity = "preference|favorite|dish"
        self.entry.report.topic_value = "risotto"
        self.d = self.entry.to_dict()

    def test_all_keys_present(self) -> None:
        expected = {"record_id", "interaction_id", "text", "token_count", "timestamp",
                    "omega", "status", "vector", "report", "provenance", "lineage",
                    "source_record"}
        assert set(self.d.keys()) == expected

    def test_record_id_is_stable_string(self) -> None:
        assert self.d["record_id"] == "record:7"

    def test_interaction_id_is_int(self) -> None:
        assert isinstance(self.d["interaction_id"], int)

    def test_interaction_id_value(self) -> None:
        assert self.d["interaction_id"] == 7

    def test_text_preserved(self) -> None:
        assert self.d["text"] == "Alice booked three tickets to Paris."

    def test_token_count_matches(self) -> None:
        assert self.d["token_count"] == self.entry.token_count

    def test_timestamp_matches(self) -> None:
        assert self.d["timestamp"] == self.entry.timestamp

    def test_omega_matches(self) -> None:
        assert self.d["omega"] == pytest.approx(0.87)

    def test_status_is_string(self) -> None:
        assert isinstance(self.d["status"], str)

    def test_status_value_correct(self) -> None:
        assert self.d["status"] == SurvivalStatus.HEALTHY.value

    def test_vector_is_list(self) -> None:
        assert isinstance(self.d["vector"], list)

    def test_vector_not_ndarray(self) -> None:
        assert not isinstance(self.d["vector"], np.ndarray)

    def test_vector_values_match(self) -> None:
        assert self.d["vector"] == pytest.approx(self.entry.vector.tolist())

    def test_vector_length_matches_dim(self) -> None:
        assert len(self.d["vector"]) == 8

    def test_report_is_dict(self) -> None:
        assert isinstance(self.d["report"], dict)

    def test_provenance_is_dict(self) -> None:
        assert isinstance(self.d["provenance"], dict)

    def test_provenance_values_match_report(self) -> None:
        assert self.d["provenance"]["role"] == "assistant"
        assert self.d["provenance"]["source_turn"] == 2
        assert self.d["provenance"]["derived_from_model"] is True

    def test_source_record_lineage_defaults_to_empty_relationships(self) -> None:
        assert self.d["lineage"] == {
            "supersedes": [],
            "conflicts_with": [],
            "corrects": [],
            "invalidates": [],
        }
        assert self.d["source_record"]["lineage"] == self.d["lineage"]

    def test_source_record_is_exposed_separately(self) -> None:
        assert self.d["source_record"]["record_id"] == "record:7"
        assert self.d["source_record"]["interaction_id"] == 7
        assert self.d["source_record"]["role"] == "assistant"
        assert self.d["source_record"]["text"] == "Alice booked three tickets to Paris."

    def test_source_record_serialises_all_signals(self) -> None:
        assert self.d["source_record"]["signals"] == {
            "is_current_state": True,
            "is_past_state": False,
            "is_preference": True,
            "is_constraint": False,
            "is_correction": False,
            "has_negation": False,
            "has_replacement": False,
            "operational_weight": 0.0,
            "personal_relevance": 0.0,
            "quantitative_relevance": 0.0,
            "task_relevance": 0.0,
            "temporal_markers": [],
            "cue_phrases": ["my favorite"],
        }

    def test_source_record_serialises_topic_fields(self) -> None:
        assert self.d["source_record"]["topic_identity"] == "preference|favorite|dish"
        assert self.d["source_record"]["topic_value"] == "risotto"

    def test_source_record_can_carry_structured_lineage(self) -> None:
        self.entry.lineage = MemoryLineage(
            supersedes=["record:1"],
            invalidates=["record:2"],
        )
        d = self.entry.to_dict()
        assert d["source_record"]["lineage"] == {
            "supersedes": ["record:1"],
            "conflicts_with": [],
            "corrects": [],
            "invalidates": ["record:2"],
        }

    def test_report_contains_survival_score(self) -> None:
        assert "survival_score" in self.d["report"]
        assert self.d["report"]["survival_score"] == pytest.approx(0.87)

    def test_result_is_json_serialisable(self) -> None:
        """json.dumps must not raise."""
        dumped = json.dumps(self.d)
        assert isinstance(dumped, str)

    def test_unscored_entry_omega_is_zero(self) -> None:
        """Entry with survival_score=None → omega property returns 0.0."""
        report = AnalysisReport(
            info_density=0.3, sentiment_abs=0.0, entity_count=0,
            is_system_prompt=False, latency_ms=1.0,
            survival_score=None, status=None,
        )
        entry = MemoryEntry(
            interaction_id=0, text="x", report=report,
            vector=_unit_vector(8, 0), token_count=1, timestamp=1.0,
        )
        d = entry.to_dict()
        assert d["omega"] == 0.0
        assert d["status"] is None


# ===========================================================================
# FileLTMHook — construction
# ===========================================================================

class TestFileLTMHookConstruction:
    """FileLTMHook __init__: directory creation, path property."""

    def test_parent_directory_created(self, tmp_path: Path) -> None:
        archive = tmp_path / "sub" / "archive.jsonl"
        hook = FileLTMHook(archive)
        assert archive.parent.is_dir()

    def test_deeply_nested_directory_created(self, tmp_path: Path) -> None:
        archive = tmp_path / "a" / "b" / "c" / "archive.jsonl"
        FileLTMHook(archive)
        assert archive.parent.is_dir()

    def test_file_not_created_before_first_archive(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        FileLTMHook(archive)
        assert not archive.exists()

    def test_path_property_returns_resolved_path(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        hook = FileLTMHook(archive)
        assert hook.path == archive

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        archive = str(tmp_path / "archive.jsonl")
        hook = FileLTMHook(archive)
        assert isinstance(hook.path, Path)

    def test_satisfies_ltm_hook_protocol(self, tmp_path: Path) -> None:
        hook = FileLTMHook(tmp_path / "archive.jsonl")
        assert isinstance(hook, LTMHook)


# ===========================================================================
# FileLTMHook — archive()
# ===========================================================================

class TestFileLTMHookArchive:
    """FileLTMHook.archive(): raw-record archival correctness."""

    def _hook(self, tmp_path: Path) -> tuple[FileLTMHook, Path]:
        archive = tmp_path / "archive.jsonl"
        return FileLTMHook(archive), archive

    def test_creates_file_on_first_call(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry())
        assert archive.exists()

    def test_one_call_writes_one_line(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry())
        lines = archive.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry(seed=0))
        hook.archive(_make_entry(seed=1))
        for line in archive.read_text(encoding="utf-8").splitlines():
            parsed = json.loads(line)
            assert isinstance(parsed, dict)

    def test_two_calls_produce_two_lines(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry(seed=0))
        hook.archive(_make_entry(seed=1))
        lines = archive.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_appends_does_not_overwrite(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        e1 = _make_entry(text="first", seed=0, interaction_id=0)
        e2 = _make_entry(text="second", seed=1, interaction_id=1)
        hook.archive(e1)
        hook.archive(e2)
        lines = archive.read_text(encoding="utf-8").splitlines()
        first_record = json.loads(lines[0])
        assert first_record["text"] == "first"

    def test_raw_record_fields_present(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry(interaction_id=42, text="the quick brown fox"))
        record = json.loads(archive.read_text(encoding="utf-8"))
        assert set(record) == {
            "record_id",
            "interaction_id",
            "role",
            "text",
            "created_at",
            "provenance",
        }
        assert record["record_id"] == "record:42"
        assert record["interaction_id"] == 42
        assert record["text"] == "the quick brown fox"

    def test_role_and_provenance_are_preserved(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        entry = _make_entry(text="remember this", interaction_id=7)
        hook.archive(entry)
        record = json.loads(archive.read_text(encoding="utf-8"))
        assert record["role"] == "assistant"
        assert record["provenance"]["role"] == "assistant"
        assert record["provenance"]["source_turn"] == 2

    def test_non_ascii_text_preserved(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        hook.archive(_make_entry(text="Ångström ≈ 1e-10 m — 日本語"))
        raw = archive.read_text(encoding="utf-8")
        record = json.loads(raw)
        assert record["text"] == "Ångström ≈ 1e-10 m — 日本語"
        assert "\\u" not in raw or "Ångström" in raw

    def test_lines_in_insertion_order(self, tmp_path: Path) -> None:
        hook, archive = self._hook(tmp_path)
        for i in range(5):
            hook.archive(_make_entry(interaction_id=i, seed=i))
        lines = archive.read_text(encoding="utf-8").splitlines()
        ids = [json.loads(l)["interaction_id"] for l in lines]
        assert ids == list(range(5))



# ===========================================================================
# LTMSettings
# ===========================================================================

class TestLTMSettings:
    """LTMSettings dataclass defaults and immutability."""

    def test_default_storage_type(self) -> None:
        assert LTMSettings().storage_type == "file"

    def test_default_storage_path(self) -> None:
        assert LTMSettings().storage_path == "data/ltm_archive.jsonl"

    def test_default_enabled(self) -> None:
        assert LTMSettings().enabled is True

    def test_is_frozen(self) -> None:
        s = LTMSettings()
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.enabled = False  # type: ignore[misc]

    def test_custom_values(self) -> None:
        s = LTMSettings(storage_type="null", storage_path="/tmp/x.jsonl", enabled=False)
        assert s.storage_type == "null"
        assert s.storage_path == "/tmp/x.jsonl"
        assert s.enabled is False


# ===========================================================================
# DMFConfig.ltm — defaults and TOML loading
# ===========================================================================

class TestDMFConfigLTM:
    """DMFConfig carries LTMSettings; load_dmf_config parses [ltm]."""

    def test_default_dmf_config_has_ltm(self) -> None:
        cfg = DMFConfig()
        assert isinstance(cfg.ltm, LTMSettings)

    def test_default_ltm_enabled(self) -> None:
        assert DMFConfig().ltm.enabled is True

    def test_default_ltm_storage_type(self) -> None:
        assert DMFConfig().ltm.storage_type == "file"

    def test_load_dmf_config_parses_ltm_section(self) -> None:
        # Module 7: TOML now defaults to storage_type="chroma" for active recall.
        cfg = load_dmf_config()
        assert cfg.ltm.storage_type == "chroma"
        assert cfg.ltm.storage_path == "data/ltm_archive.jsonl"
        assert cfg.ltm.chroma_path == "data/ltm_chroma"
        assert cfg.ltm.recall_limit == 5
        assert cfg.ltm.distance_threshold == 0.7
        assert cfg.ltm.enabled is True

    def test_load_dmf_config_custom_toml(self, tmp_path: Path) -> None:
        """A custom TOML with [ltm] disabled must be parsed correctly."""
        toml = tmp_path / "test_settings.toml"
        toml.write_text(
            "[ltm]\n"
            'storage_type = "file"\n'
            'storage_path = "/tmp/custom.jsonl"\n'
            "enabled = false\n",
            encoding="utf-8",
        )
        cfg = load_dmf_config(toml)
        assert cfg.ltm.storage_path == "/tmp/custom.jsonl"
        assert cfg.ltm.enabled is False

    def test_load_dmf_config_missing_ltm_section_uses_defaults(
        self, tmp_path: Path
    ) -> None:
        """TOML without [ltm] must fall back to LTMSettings defaults."""
        toml = tmp_path / "minimal.toml"
        toml.write_text("# no ltm section\n", encoding="utf-8")
        cfg = load_dmf_config(toml)
        assert cfg.ltm == LTMSettings()


# ===========================================================================
# TemporalMemory.from_dmf_config — LTM hook resolution
# ===========================================================================

class TestFromDMFConfigLTMResolution:
    """from_dmf_config wires up the correct LTMHook from DMFConfig.ltm."""

    def _dmf_cfg(
        self,
        tmp_path: Path,
        enabled: bool = True,
        storage_type: str = "file",
    ) -> DMFConfig:
        return DMFConfig(
            ltm=LTMSettings(
                storage_type=storage_type,
                storage_path=str(tmp_path / "archive.jsonl"),
                enabled=enabled,
            )
        )

    def test_file_hook_created_when_enabled(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=True)
        tm = TemporalMemory.from_dmf_config(cfg)
        assert isinstance(tm._ltm_hook, FileLTMHook)

    def test_file_hook_path_matches_config(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=True)
        tm = TemporalMemory.from_dmf_config(cfg)
        assert tm._ltm_hook.path == Path(cfg.ltm.storage_path)  # type: ignore[attr-defined]

    def test_null_hook_used_when_disabled(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=False)
        tm = TemporalMemory.from_dmf_config(cfg)
        assert isinstance(tm._ltm_hook, NullLTMHook)

    def test_explicit_hook_overrides_config(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=True)
        explicit = NullLTMHook()
        tm = TemporalMemory.from_dmf_config(cfg, ltm_hook=explicit)
        assert tm._ltm_hook is explicit

    def test_unknown_storage_type_raises(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=True, storage_type="lancedb")
        with pytest.raises(ValueError, match="Unsupported ltm.storage_type at runtime"):
            TemporalMemory.from_dmf_config(cfg)

    def test_null_hook_when_storage_type_is_explicit_null(self, tmp_path: Path) -> None:
        cfg = self._dmf_cfg(tmp_path, enabled=True, storage_type="null")
        tm = TemporalMemory.from_dmf_config(cfg)
        assert isinstance(tm._ltm_hook, NullLTMHook)

    def test_qdrant_hook_created_when_storage_type_is_qdrant(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        captured: dict[str, object] = {}

        class FakeQdrantLTMHook:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(ltm_hooks, "QdrantLTMHook", FakeQdrantLTMHook)
        cfg = self._dmf_cfg(tmp_path, enabled=True, storage_type="qdrant")

        tm = TemporalMemory.from_dmf_config(cfg)

        assert isinstance(tm._ltm_hook, FakeQdrantLTMHook)
        assert captured["collection_name"] == cfg.ltm.collection_name
        assert captured["distance_threshold"] == cfg.ltm.distance_threshold

    def test_explicit_hook_overrides_qdrant_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(
            ltm_hooks,
            "QdrantLTMHook",
            lambda **kwargs: pytest.fail("unexpected Qdrant construction"),
        )
        cfg = self._dmf_cfg(tmp_path, enabled=True, storage_type="qdrant")
        explicit = NullLTMHook()

        tm = TemporalMemory.from_dmf_config(cfg, ltm_hook=explicit)

        assert tm._ltm_hook is explicit


# ===========================================================================
# Integration — eviction → JSONL archive
# ===========================================================================

class TestLTMPersistenceIntegration:
    """End-to-end: TemporalMemory eviction writes raw records to JSONL."""

    def _make_hook_and_tm(
        self,
        tmp_path: Path,
        budget: int = 1,
    ) -> tuple[FileLTMHook, TemporalMemory]:
        archive = tmp_path / "archive.jsonl"
        hook = FileLTMHook(archive)
        tm = _budget_tm(budget=budget, hook=hook)
        return hook, tm

    def _read_records(self, hook: FileLTMHook) -> list[dict]:
        if not hook.path.exists():
            return []
        lines = hook.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(l) for l in lines if l.strip()]

    def test_evicted_entry_appears_in_archive(self, tmp_path: Path) -> None:
        hook, tm = self._make_hook_and_tm(tmp_path, budget=1)
        tm.add_interaction("first", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("second", _make_report(omega=0.15), _unit_vector(8, 1))
        records = self._read_records(hook)
        assert len(records) >= 1

    def test_evicted_text_in_archive(self, tmp_path: Path) -> None:
        hook, tm = self._make_hook_and_tm(tmp_path, budget=1)
        tm.add_interaction("evict me", _make_report(omega=0.15), _unit_vector(8, 0))
        tm.add_interaction("keep me", _make_report(omega=0.15), _unit_vector(8, 1))
        records = self._read_records(hook)
        texts = [r["text"] for r in records]
        assert "evict me" in texts

    def test_evicted_raw_record_id_matches_entry(self, tmp_path: Path) -> None:
        hook, tm = self._make_hook_and_tm(tmp_path, budget=1)
        tm.add_interaction("msg", _make_report(omega=0.22), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.22), _unit_vector(8, 1))
        records = self._read_records(hook)
        assert any(r["record_id"] == "record:0" for r in records)

    def test_evicted_archive_keeps_provenance(self, tmp_path: Path) -> None:
        hook, tm = self._make_hook_and_tm(tmp_path, budget=1)
        tm.add_interaction("msg", _make_report(omega=0.22), _unit_vector(8, 0))
        tm.add_interaction("x", _make_report(omega=0.22), _unit_vector(8, 1))
        records = self._read_records(hook)
        assert records[0]["provenance"]["role"] == "assistant"

    def test_multiple_evictions_all_archived(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        hook = FileLTMHook(archive)
        cfg = DecayConfig(token_budget=2, pruning_frequency=999_999)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        for i in range(4):
            tm.add_interaction(f"message {i}", _make_report(omega=0.15), _unit_vector(8, i))
        records = self._read_records(hook)
        assert len(records) >= 3

    def test_archive_preserves_insertion_order(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        hook = FileLTMHook(archive)
        cfg = DecayConfig(token_budget=2, pruning_frequency=999_999)
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        texts = [f"msg_{i}" for i in range(4)]
        for i, text in enumerate(texts):
            tm.add_interaction(text, _make_report(omega=0.15), _unit_vector(8, i))
        records = self._read_records(hook)
        archived_ids = [r["interaction_id"] for r in records]
        assert archived_ids == sorted(archived_ids)

    def test_periodic_cleanup_eviction_archived(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        hook = FileLTMHook(archive)
        cfg = DecayConfig(
            lambda_decay=2.0,
            inertia_strength=0.5,
            hard_kill_threshold=0.4,
            token_budget=999_999,
            pruning_frequency=3,
        )
        tm = TemporalMemory(decay_config=cfg, vector_config=_VECTOR_CFG, ltm_hook=hook)
        tm.add_interaction("old message", _make_report(omega=0.45), _unit_vector(8, 0))
        tm.add_interaction("filler 1", _make_report(omega=0.45), _unit_vector(8, 1))
        tm.add_interaction("filler 2", _make_report(omega=0.45), _unit_vector(8, 2))
        records = self._read_records(hook)
        assert any(r["text"] == "old message" for r in records)

    def test_archive_is_append_only(self, tmp_path: Path) -> None:
        archive = tmp_path / "archive.jsonl"
        archive.write_text('{"sentinel": true, "interaction_id": -1}\n', encoding="utf-8")
        hook = FileLTMHook(archive)
        hook.archive(_make_entry(interaction_id=0))
        lines = archive.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first.get("sentinel") is True


# ===========================================================================
# FileLTMHook — read_all()
# ===========================================================================

class TestFileLTMHookReadAll:
    """FileLTMHook.read_all(): reading back archived raw records."""

    def _hook(self, tmp_path: Path) -> tuple[FileLTMHook, Path]:
        archive = tmp_path / "archive.jsonl"
        return FileLTMHook(archive), archive

    def test_read_all_returns_empty_when_file_does_not_exist(self, tmp_path: Path) -> None:
        hook, _ = self._hook(tmp_path)
        assert hook.read_all() == []

    def test_read_all_returns_all_archived_records(self, tmp_path: Path) -> None:
        from dmf.models.raw_ltm import RawLTMRecord
        hook, _ = self._hook(tmp_path)
        e1 = _make_entry(text="first", interaction_id=0)
        e2 = _make_entry(text="second", interaction_id=1)
        hook.archive(e1)
        hook.archive(e2)

        result = hook.read_all()

        assert len(result) == 2
        assert all(isinstance(r, RawLTMRecord) for r in result)
        assert result[0].text == "first"
        assert result[1].text == "second"

    def test_read_all_preserves_insertion_order(self, tmp_path: Path) -> None:
        hook, _ = self._hook(tmp_path)
        for i in range(5):
            hook.archive(_make_entry(interaction_id=i, seed=i))

        result = hook.read_all()

        assert [r.interaction_id for r in result] == list(range(5))

    def test_read_all_round_trips_record_id_and_role(self, tmp_path: Path) -> None:
        hook, _ = self._hook(tmp_path)
        hook.archive(_make_entry(interaction_id=42))

        result = hook.read_all()

        assert result[0].record_id == "record:42"
        assert result[0].role == "assistant"
