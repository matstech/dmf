"""Deterministic unit tests for the opt-in local LTM benchmark utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrationtest.run_ltm_benchmark import (
    BACKEND_CHROMA,
    BACKEND_QDRANT,
    BenchmarkError,
    OllamaClient,
    build_benchmark_config,
    client_version_for_backend,
    build_ollama_messages,
    load_dataset,
    load_report,
    new_report,
    parse_args,
    parse_ollama_models,
    render_summary_table,
    score_text,
    summarize_report,
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


def test_ollama_chat_disables_model_thinking() -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        content = b'{"message": {"content": "ok"}}'

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"message": {"content": "ok"}}

    class FakeClient:
        def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    client = OllamaClient("http://localhost:11434", "qwen2.5:0.5b")
    client._client = FakeClient()  # type: ignore[assignment]

    response, latency_ms = client.chat("Domanda?", "Contesto")

    assert response == "ok"
    assert latency_ms >= 0
    assert captured["path"] == "/api/chat"
    assert captured["json"]["think"] is False  # type: ignore[index]
    assert captured["json"]["stream"] is False  # type: ignore[index]


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


def test_summary_table_is_computed_from_report_json(tmp_path: Path) -> None:
    report = {
        "status": "complete",
        "backend": BACKEND_QDRANT,
        "benchmark_id": "bench-test",
        "ollama": {"model": "qwen2.5:0.5b", "base_url": "http://localhost:11434"},
        "ltm": {
            "backend": BACKEND_QDRANT,
            "collection": "dmf_benchmark_test",
            "count_after_seed": 4,
        },
        "seed": {"seed_turn_count": 3, "filler_turn_count": 1, "count_after_seed": 4},
        "aggregate": {
            "case_count": 2,
            "completed_case_count": 2,
            "failed_case_count": 0,
            "retrieval_success_count": 1,
            "retrieval_success_rate": 0.5,
            "mean_answer_score": 0.625,
            "total_ollama_latency_ms": 300.0,
            "mean_ollama_latency_ms": 150.0,
        },
        "cases": [
            {
                "id": "case-a",
                "category": "preference",
                "error": None,
                "retrieval_success": True,
                "ollama_latency_ms": 100.0,
                "retrieval_score": {
                    "matched_groups": [["alpha"], ["beta"]],
                    "missing_groups": [],
                    "promoted_obsolete_groups": [],
                },
                "answer_score": {
                    "score": 1.0,
                    "matched_groups": [["alpha"]],
                    "missing_groups": [],
                    "promoted_obsolete_groups": [],
                },
            },
            {
                "id": "case-b",
                "category": "constraint",
                "error": None,
                "retrieval_success": False,
                "ollama_latency_ms": 200.0,
                "retrieval_score": {
                    "matched_groups": [["gamma"]],
                    "missing_groups": [["delta"]],
                    "promoted_obsolete_groups": [["old"]],
                },
                "answer_score": {
                    "score": 0.25,
                    "matched_groups": [["gamma"]],
                    "missing_groups": [["delta"]],
                    "promoted_obsolete_groups": [["old"]],
                },
            },
        ],
        "errors": [],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    summary = summarize_report(load_report(path))
    table = render_summary_table(summary)

    assert summary["retrieval_precision"] == pytest.approx(0.75)
    assert summary["retrieval_recall"] == pytest.approx(0.75)
    assert summary["answer_precision"] == pytest.approx(2 / 3)
    assert summary["answer_recall"] == pytest.approx(2 / 3)
    assert summary["min_ollama_latency_ms"] == 100.0
    assert summary["p50_ollama_latency_ms"] == 150.0
    assert summary["max_ollama_latency_ms"] == 200.0
    assert "Benchmark aggregate results" in table
    assert "Retrieval term precision" in table
    assert "75.00%" in table
    assert "Retrieval success overall" in table
    assert "Answer term recall" in table
    assert "66.67%" in table
    assert "Ollama latency p95" in table
    assert "Throughput" in table


def test_summary_distinguishes_completed_and_overall_retrieval_rates() -> None:
    report = {
        "status": "partial_failure",
        "backend": BACKEND_QDRANT,
        "benchmark_id": "bench-test",
        "ollama": {"model": "qwen2.5:0.5b"},
        "ltm": {"collection": "dmf_benchmark_test"},
        "seed": {"count_after_seed": 4},
        "aggregate": {
            "case_count": 3,
            "completed_case_count": 2,
            "failed_case_count": 1,
            "retrieval_success_count": 1,
            "retrieval_success_rate": 1 / 3,
        },
        "cases": [
            {
                "id": "case-a",
                "category": "preference",
                "error": None,
                "retrieval_success": True,
                "retrieval_score": {"matched_groups": [["a"]], "missing_groups": []},
                "answer_score": {"matched_groups": [["a"]], "missing_groups": []},
            },
            {
                "id": "case-b",
                "category": "constraint",
                "error": None,
                "retrieval_success": False,
                "retrieval_score": {"matched_groups": [], "missing_groups": [["b"]]},
                "answer_score": {"matched_groups": [], "missing_groups": [["b"]]},
            },
            {
                "id": "case-c",
                "category": "state",
                "error": "BenchmarkError: empty content",
                "retrieval_success": False,
            },
        ],
    }

    summary = summarize_report(report)
    table = render_summary_table(summary)

    assert summary["retrieval_success_rate"] == pytest.approx(0.5)
    assert summary["retrieval_success_rate_overall"] == pytest.approx(1 / 3)
    assert "50.00%" in table
    assert "33.33%" in table


def test_load_report_rejects_non_object_json(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(BenchmarkError, match="JSON object"):
        load_report(path)
