"""Deterministic unit tests for the opt-in local LTM benchmark utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrationtest.run_ltm_benchmark import (
    BACKEND_CHROMA,
    BACKEND_QDRANT,
    BenchmarkError,
    build_benchmark_config,
    client_version_for_backend,
    build_ollama_messages,
    load_dataset,
    new_report,
    parse_args,
    parse_ollama_models,
    score_text,
    validate_dataset,
    validate_loopback_host,
    validate_loopback_url,
)
from dmf.utils.config_loader import DMFConfig

DATASET_PATH = Path(__file__).parents[1] / "integrationtest" / "benchmark_cases.json"


def test_versioned_dataset_has_exactly_ten_valid_cases() -> None:
    dataset = load_dataset(DATASET_PATH)

    assert dataset.schema_version == 1
    assert dataset.benchmark_id == "dmf-ltm-local-v1"
    assert len(dataset.cases) == 10
    assert len({case.case_id for case in dataset.cases}) == 10


def test_dataset_rejects_more_than_ten_cases() -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["cases"].append(dict(payload["cases"][0], id="case_11"))

    with pytest.raises(BenchmarkError, match="between 1 and 10"):
        validate_dataset(payload)


def test_dataset_rejects_non_boolean_provenance_flags() -> None:
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    payload["seed"][0]["is_constraint"] = "false"

    with pytest.raises(BenchmarkError, match="must be boolean"):
        validate_dataset(payload)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:11434", "http://localhost:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        ("http://[::1]:11434", "http://[::1]:11434"),
    ],
)
def test_validate_loopback_url_accepts_only_sanitized_local_urls(
    value: str,
    expected: str,
) -> None:
    assert validate_loopback_url(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://example.com:11434",
        "http://user:secret@localhost:11434",
        "http://localhost:11434/api",
        "file://localhost/tmp/ollama",
    ],
)
def test_validate_loopback_url_rejects_remote_or_unsafe_urls(value: str) -> None:
    with pytest.raises(BenchmarkError):
        validate_loopback_url(value)


def test_validate_loopback_host_rejects_non_local_chroma() -> None:
    with pytest.raises(BenchmarkError, match="loopback-only"):
        validate_loopback_host("chroma.example.com", field_name="CHROMA_HOST")


def test_score_text_matches_aliases_and_normalizes_accents() -> None:
    result = score_text(
        "La riunione è MARTEDÌ alle 14.",
        (("martedi",), ("14",)),
    )

    assert result["score"] == 1.0
    assert result["missing_groups"] == []


def test_score_text_penalizes_obsolete_value_promoted_as_current() -> None:
    result = score_text(
        "Il budget è 1500 oppure 1200.",
        (("1500",),),
        (("1200",),),
    )

    assert result["coverage"] == 1.0
    assert result["obsolete_penalty"] == 0.5
    assert result["score"] == 0.5


def test_score_text_does_not_penalize_explicitly_rejected_obsolete_value() -> None:
    result = score_text(
        "Il budget corrente è 1500, non 1200.",
        (("1500",),),
        (("1200",),),
    )

    assert result["score"] == 1.0
    assert result["promoted_obsolete_groups"] == []


def test_ollama_messages_are_stateless_and_contain_only_current_inputs() -> None:
    messages = build_ollama_messages("Domanda corrente?", "Contesto recuperato")
    serialized = json.dumps(messages, ensure_ascii=False)

    assert len(messages) == 2
    assert "Domanda corrente?" in serialized
    assert "Contesto recuperato" in serialized
    assert "seed history" not in serialized


def test_parse_ollama_models_validates_shape() -> None:
    assert parse_ollama_models({"models": [{"name": "qwen2.5:0.5b"}]}) == {
        "qwen2.5:0.5b"
    }
    with pytest.raises(BenchmarkError, match="invalid response"):
        parse_ollama_models({"models": "invalid"})


def test_benchmark_config_forces_server_and_bounded_pressure() -> None:
    config = build_benchmark_config(
        DMFConfig(),
        collection_name="dmf_benchmark_test",
        chroma_host="localhost",
        chroma_port=8000,
    )

    assert config.ltm.chroma_mode == "server"
    assert config.ltm.collection_name == "dmf_benchmark_test"
    assert config.ltm.cards_enabled is False
    assert config.capacity.token_budget == 48
    assert config.capacity.pruning_frequency_x == 1
    assert config.tiers.healthy_min == 1.0
    assert config.decay.lambda_base == 1.0
    assert config.decay.inertia_strength == 0.0


def test_parse_args_defaults_to_chroma_backend() -> None:
    args = parse_args([])

    assert args.backend == BACKEND_CHROMA


def test_parse_args_accepts_qdrant_backend() -> None:
    args = parse_args(["--backend", "qdrant"])

    assert args.backend == BACKEND_QDRANT


def test_parse_args_rejects_unknown_backend() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--backend", "redis"])


def test_benchmark_config_for_qdrant_uses_memory_without_chroma_server() -> None:
    config = build_benchmark_config(
        DMFConfig(),
        collection_name="dmf_benchmark_test",
        backend=BACKEND_QDRANT,
        chroma_host="ignored-host",
        chroma_port=6553,
    )

    assert config.ltm.storage_type == "qdrant"
    assert config.ltm.qdrant_mode == "memory"
    assert config.ltm.collection_name == "dmf_benchmark_test"
    assert config.ltm.cards_enabled is False
    assert config.ltm.chroma_mode == DMFConfig().ltm.chroma_mode
    assert config.capacity.token_budget == 48


def test_benchmark_report_contains_backend_and_client_version() -> None:
    dataset = load_dataset(DATASET_PATH)

    report = new_report(
        dataset,
        "qwen2.5:0.5b",
        "http://localhost:11434",
        backend=BACKEND_QDRANT,
    )
    report["ltm"] = {
        "backend": BACKEND_QDRANT,
        "client_version": client_version_for_backend(BACKEND_QDRANT),
        "collection": "dmf_benchmark_test",
        "count_after_seed": 3,
    }

    assert report["backend"] == BACKEND_QDRANT
    assert report["ltm"]["backend"] == BACKEND_QDRANT
    assert report["ltm"]["client_version"]
    assert report["ltm"]["collection"] == "dmf_benchmark_test"
    assert report["ltm"]["count_after_seed"] == 3
