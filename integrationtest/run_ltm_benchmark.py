"""Run the opt-in local DMF LTM benchmark against Chroma and Ollama."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dmf.analysis import EmbeddingEngine, NLPEngine, ScoringEngine  # noqa: E402
from dmf.memory import Memory, TemporalMemory  # noqa: E402
from dmf.memory.ltm_hooks import ChromaLTMHook  # noqa: E402
from dmf.memory.ltm_hooks.chroma_client import (  # noqa: E402
    ChromaConnectionConfig,
    ChromaConnectionMode,
)
from dmf.models.analysis import InteractionProvenance  # noqa: E402
from dmf.runtime.pipeline import InteractionPipeline  # noqa: E402
from dmf.utils.config import NLPConfig, VectorConfig  # noqa: E402
from dmf.utils.config_loader import DMFConfig, load_dmf_config  # noqa: E402
from dmf.utils.constants import DEFAULT_EMBEDDING_CACHE_DIR  # noqa: E402

DATASET_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
SCORER_VERSION = "term-coverage-v1"
DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
MAX_CASES = 10
MAX_RESPONSE_CHARS = 4_096
MAX_RESPONSE_BYTES = 1_000_000
MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SAFE_OBSOLETE_PATTERNS = (
    r"\bnon\s+(?:e\s+|era\s+|a\s+)?{term}\b",
    r"\bnot\s+{term}\b",
    r"\b{term}\b.{{0,28}}\b(?:precedente|passato|concluso|superato|vecchio)\b",
    r"\b(?:prima|precedentemente|in passato)\b.{{0,28}}\b{term}\b",
)


class BenchmarkError(RuntimeError):
    """Expected local benchmark configuration or infrastructure failure."""


TermGroups = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class SeedTurn:
    """One deterministic interaction ingested before benchmark questions."""

    seed_id: str
    text: str
    role: str
    is_user_correction: bool = False
    is_preference_update: bool = False
    is_constraint: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    """One retrieval and response-quality benchmark case."""

    case_id: str
    category: str
    question: str
    expected_answer: str
    evidence_terms: TermGroups
    answer_terms: TermGroups
    obsolete_terms: TermGroups


@dataclass(frozen=True)
class FillerConfig:
    """Bounded deterministic pressure configuration."""

    max_turns: int
    template: str


@dataclass(frozen=True)
class BenchmarkDataset:
    """Validated benchmark dataset."""

    schema_version: int
    benchmark_id: str
    seed: tuple[SeedTurn, ...]
    filler: FillerConfig
    cases: tuple[BenchmarkCase, ...]


@dataclass
class BenchmarkRuntime:
    """Public DMF components used by one isolated benchmark run."""

    hook: ChromaLTMHook
    temporal_memory: TemporalMemory
    memory: Memory
    pipeline: InteractionPipeline
    scoring_engine: ScoringEngine
    embedding_engine: EmbeddingEngine


def normalize_text(value: str) -> str:
    """Normalize text for transparent accent-insensitive term matching."""
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(without_marks.split())


def validate_loopback_host(host: str, *, field_name: str = "host") -> str:
    """Return a normalized loopback host or reject non-local networking."""
    normalized = host.strip().lower().strip("[]")
    if normalized not in LOOPBACK_HOSTS:
        raise BenchmarkError(
            f"{field_name} must be loopback-only: localhost, 127.0.0.1, or ::1"
        )
    return normalized


def validate_loopback_url(value: str) -> str:
    """Validate and sanitize the Ollama base URL."""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise BenchmarkError("OLLAMA_BASE_URL has an invalid port") from exc

    if parsed.scheme not in {"http", "https"}:
        raise BenchmarkError("OLLAMA_BASE_URL must use http or https")
    if parsed.username or parsed.password:
        raise BenchmarkError("OLLAMA_BASE_URL must not contain credentials")
    if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise BenchmarkError("OLLAMA_BASE_URL must not contain path, query, or fragment")
    host = validate_loopback_host(parsed.hostname or "", field_name="OLLAMA_BASE_URL host")
    rendered_host = f"[{host}]" if ":" in host else host
    rendered_port = f":{port}" if port is not None else ""
    return f"{parsed.scheme}://{rendered_host}{rendered_port}"


def validate_model_name(value: str) -> str:
    """Reject empty, unbounded, or control-character model identifiers."""
    model = value.strip()
    if not MODEL_PATTERN.fullmatch(model):
        raise BenchmarkError("OLLAMA_MODEL is not a valid local model identifier")
    return model


def parse_port(value: str, *, field_name: str) -> int:
    """Parse a TCP port from one environment variable."""
    try:
        port = int(value)
    except ValueError as exc:
        raise BenchmarkError(f"{field_name} must be an integer") from exc
    if not 1 <= port <= 65_535:
        raise BenchmarkError(f"{field_name} must be between 1 and 65535")
    return port


def _required_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError(f"Dataset field {field_name} must be a non-empty string")
    return value.strip()


def _optional_bool(raw: dict[str, object], field_name: str, *, prefix: str) -> bool:
    value = raw.get(field_name, False)
    if not isinstance(value, bool):
        raise BenchmarkError(f"Dataset field {prefix}.{field_name} must be boolean")
    return value


def _term_groups(value: object, *, field_name: str, allow_empty: bool = False) -> TermGroups:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise BenchmarkError(f"Dataset field {field_name} must be {qualifier} of alias groups")
    groups: list[tuple[str, ...]] = []
    for index, group in enumerate(value):
        if not isinstance(group, list) or not group:
            raise BenchmarkError(f"Dataset field {field_name}[{index}] must be non-empty")
        aliases = tuple(
            _required_string(alias, field_name=f"{field_name}[{index}]")
            for alias in group
        )
        groups.append(aliases)
    return tuple(groups)


def validate_dataset(payload: object) -> BenchmarkDataset:
    """Validate the versioned local dataset and enforce the hard case limit."""
    if not isinstance(payload, dict):
        raise BenchmarkError("Dataset root must be a JSON object")
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise BenchmarkError(
            f"Unsupported dataset schema_version: {payload.get('schema_version')!r}"
        )
    benchmark_id = _required_string(payload.get("benchmark_id"), field_name="benchmark_id")

    raw_seed = payload.get("seed")
    if not isinstance(raw_seed, list) or not raw_seed:
        raise BenchmarkError("Dataset seed must be a non-empty list")
    seed: list[SeedTurn] = []
    seed_ids: set[str] = set()
    for index, raw in enumerate(raw_seed):
        if not isinstance(raw, dict):
            raise BenchmarkError(f"Dataset seed[{index}] must be an object")
        seed_id = _required_string(raw.get("id"), field_name=f"seed[{index}].id")
        if seed_id in seed_ids:
            raise BenchmarkError(f"Duplicate seed id: {seed_id}")
        seed_ids.add(seed_id)
        role = _required_string(raw.get("role"), field_name=f"seed[{index}].role")
        if role not in {"user", "assistant"}:
            raise BenchmarkError(f"Dataset seed[{index}].role must be user or assistant")
        seed.append(
            SeedTurn(
                seed_id=seed_id,
                text=_required_string(raw.get("text"), field_name=f"seed[{index}].text"),
                role=role,
                is_user_correction=_optional_bool(
                    raw, "is_user_correction", prefix=f"seed[{index}]"
                ),
                is_preference_update=_optional_bool(
                    raw, "is_preference_update", prefix=f"seed[{index}]"
                ),
                is_constraint=_optional_bool(
                    raw, "is_constraint", prefix=f"seed[{index}]"
                ),
            )
        )

    raw_filler = payload.get("filler")
    if not isinstance(raw_filler, dict):
        raise BenchmarkError("Dataset filler must be an object")
    max_turns = raw_filler.get("max_turns")
    if not isinstance(max_turns, int) or not 1 <= max_turns <= 100:
        raise BenchmarkError("Dataset filler.max_turns must be between 1 and 100")
    template = _required_string(raw_filler.get("template"), field_name="filler.template")
    if "{index}" not in template:
        raise BenchmarkError("Dataset filler.template must contain {index}")
    try:
        template.format(index=1)
    except (KeyError, ValueError) as exc:
        raise BenchmarkError("Dataset filler.template must accept {index}") from exc

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not 1 <= len(raw_cases) <= MAX_CASES:
        raise BenchmarkError(f"Dataset must contain between 1 and {MAX_CASES} cases")
    cases: list[BenchmarkCase] = []
    case_ids: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise BenchmarkError(f"Dataset cases[{index}] must be an object")
        case_id = _required_string(raw.get("id"), field_name=f"cases[{index}].id")
        if case_id in case_ids:
            raise BenchmarkError(f"Duplicate case id: {case_id}")
        case_ids.add(case_id)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                category=_required_string(
                    raw.get("category"), field_name=f"cases[{index}].category"
                ),
                question=_required_string(
                    raw.get("question"), field_name=f"cases[{index}].question"
                ),
                expected_answer=_required_string(
                    raw.get("expected_answer"),
                    field_name=f"cases[{index}].expected_answer",
                ),
                evidence_terms=_term_groups(
                    raw.get("evidence_terms"),
                    field_name=f"cases[{index}].evidence_terms",
                ),
                answer_terms=_term_groups(
                    raw.get("answer_terms"),
                    field_name=f"cases[{index}].answer_terms",
                ),
                obsolete_terms=_term_groups(
                    raw.get("obsolete_terms", []),
                    field_name=f"cases[{index}].obsolete_terms",
                    allow_empty=True,
                ),
            )
        )

    return BenchmarkDataset(
        schema_version=DATASET_SCHEMA_VERSION,
        benchmark_id=benchmark_id,
        seed=tuple(seed),
        filler=FillerConfig(max_turns=max_turns, template=template),
        cases=tuple(cases),
    )


def load_dataset(path: Path) -> BenchmarkDataset:
    """Load and validate one UTF-8 benchmark dataset."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"Cannot load dataset {path.name}: {type(exc).__name__}") from exc
    return validate_dataset(payload)


def _term_present(normalized_text: str, alias: str) -> bool:
    normalized_alias = normalize_text(alias)
    return bool(
        re.search(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", normalized_text)
    )


def _obsolete_promoted(normalized_text: str, group: tuple[str, ...]) -> bool:
    for alias in group:
        normalized_alias = normalize_text(alias)
        if not _term_present(normalized_text, normalized_alias):
            continue
        safe = any(
            re.search(pattern.format(term=re.escape(normalized_alias)), normalized_text)
            for pattern in SAFE_OBSOLETE_PATTERNS
        )
        if not safe:
            return True
    return False


def score_text(
    text: str,
    required_groups: TermGroups,
    obsolete_groups: TermGroups = (),
) -> dict[str, object]:
    """Score alias-group coverage with an explicit obsolete-value penalty."""
    normalized = normalize_text(text)
    matched = [
        list(group)
        for group in required_groups
        if any(_term_present(normalized, alias) for alias in group)
    ]
    missing = [
        list(group)
        for group in required_groups
        if not any(_term_present(normalized, alias) for alias in group)
    ]
    promoted = [
        list(group) for group in obsolete_groups if _obsolete_promoted(normalized, group)
    ]
    coverage = len(matched) / len(required_groups) if required_groups else 1.0
    obsolete_penalty = 0.5 * len(promoted) / max(1, len(obsolete_groups))
    score = round(max(0.0, min(1.0, coverage - obsolete_penalty)), 4)
    return {
        "score": score,
        "coverage": round(coverage, 4),
        "obsolete_penalty": round(obsolete_penalty, 4),
        "matched_groups": matched,
        "missing_groups": missing,
        "promoted_obsolete_groups": promoted,
    }


def build_ollama_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build a stateless prompt containing only current question and DMF context."""
    if len(context) > 12_000:
        raise BenchmarkError("DMF context exceeds the 12000-character benchmark limit")
    return [
        {
            "role": "system",
            "content": (
                "Rispondi in modo conciso usando soltanto il contesto DMF fornito. "
                "Se il contesto non contiene la risposta, dichiaralo esplicitamente."
            ),
        },
        {
            "role": "user",
            "content": f"Contesto DMF:\n{context or '[nessuna evidenza]'}\n\nDomanda:\n{question}",
        },
    ]


def parse_ollama_models(payload: object) -> set[str]:
    """Extract locally installed model names from Ollama's tags response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise BenchmarkError("Ollama /api/tags returned an invalid response")
    names: set[str] = set()
    for item in payload["models"]:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(item["name"])
    return names


class OllamaClient:
    """Bounded direct HTTP client for a loopback-only Ollama service."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = validate_loopback_url(base_url)
        self.model = validate_model_name(model)
        timeout = httpx.Timeout(connect=3.0, read=90.0, write=10.0, pool=3.0)
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def __enter__(self) -> OllamaClient:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def preflight(self) -> None:
        """Require an already-running Ollama service and installed model."""
        try:
            response = self._client.get("/api/tags")
            response.raise_for_status()
            models = parse_ollama_models(response.json())
        except (httpx.HTTPError, ValueError, json.JSONDecodeError) as exc:
            raise BenchmarkError(
                f"Ollama preflight failed at {self.base_url}: {type(exc).__name__}"
            ) from exc
        if self.model not in models:
            raise BenchmarkError(
                f"Ollama model {self.model!r} is not installed locally; "
                "install it explicitly before running the benchmark"
            )

    def chat(self, question: str, context: str) -> tuple[str, float]:
        """Run one independent non-streaming chat request."""
        started = time.perf_counter()
        response = self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": build_ollama_messages(question, context),
                "stream": False,
                "options": {"temperature": 0, "num_predict": 128},
            },
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        response.raise_for_status()
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise BenchmarkError("Ollama response exceeds the benchmark byte limit")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise BenchmarkError("Ollama /api/chat returned an invalid response")
        content = payload["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise BenchmarkError("Ollama /api/chat returned empty content")
        content = content.strip()
        if len(content) > MAX_RESPONSE_CHARS:
            raise BenchmarkError("Ollama content exceeds the benchmark character limit")
        return content, round(latency_ms, 3)


def build_benchmark_config(
    base: DMFConfig,
    *,
    collection_name: str,
    chroma_host: str,
    chroma_port: int,
) -> DMFConfig:
    """Derive a controlled server configuration without mutating root settings."""
    return dataclasses.replace(
        base,
        decay=dataclasses.replace(
            base.decay,
            lambda_base=1.0,
            inertia_strength=0.0,
            hard_kill_threshold=0.05,
        ),
        tiers=dataclasses.replace(base.tiers, unstable_max=1.0, healthy_min=1.0),
        capacity=dataclasses.replace(
            base.capacity,
            token_budget=48,
            pruning_frequency_x=1,
        ),
        ltm=dataclasses.replace(
            base.ltm,
            enabled=True,
            storage_type="chroma",
            chroma_mode="server",
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            chroma_ssl=False,
            chroma_auth_token_env="",
            collection_name=collection_name,
            recall_limit=64,
            distance_threshold=2.0,
            cards_enabled=False,
        ),
        retrieval=dataclasses.replace(
            base.retrieval,
            raw_prefetch_k=64,
            final_recall_limit=10,
            enable_card_semantic=False,
            enable_card_symbolic=False,
            enable_raw_semantic=True,
            enable_raw_lexical=True,
        ),
    )


def require_local_embedding_cache(repo_root: Path) -> Path:
    """Fail before model construction when FastEmbed assets are not local."""
    cache = repo_root / DEFAULT_EMBEDDING_CACHE_DIR
    if not cache.is_dir() or not any(path.is_file() for path in cache.rglob("*")):
        raise BenchmarkError(
            f"Embedding cache is unavailable at {cache}; no model download is attempted"
        )
    return cache


def build_runtime(config: DMFConfig) -> BenchmarkRuntime:
    """Wire public DMF Pipeline, TemporalMemory, Memory, and server hook APIs."""
    vector_config = VectorConfig(
        model_name=config.nlp.model_name,
        vector_dim=config.nlp.vector_dim,
        cache_dir=str(REPO_ROOT / DEFAULT_EMBEDDING_CACHE_DIR),
        window_size=config.capacity.window_size,
    )
    embedding_engine = EmbeddingEngine(vector_config)
    connection = ChromaConnectionConfig(
        mode=ChromaConnectionMode.SERVER,
        host=config.ltm.chroma_host,
        port=config.ltm.chroma_port,
        ssl=False,
        tenant=config.ltm.chroma_tenant,
        database=config.ltm.chroma_database,
    )
    hook = ChromaLTMHook(
        collection_name=config.ltm.collection_name,
        distance_threshold=config.ltm.distance_threshold,
        embed_text=embedding_engine.get_embedding,
        connection=connection,
    )
    nlp_engine = NLPEngine(NLPConfig(spacy_model=config.nlp.spacy_model))
    temporal_memory = TemporalMemory.from_dmf_config(
        config,
        ltm_hook=hook,
        nlp_engine=nlp_engine,
    )
    memory = Memory.from_dmf_config(config, temporal_memory, embedding_engine)
    return BenchmarkRuntime(
        hook=hook,
        temporal_memory=temporal_memory,
        memory=memory,
        pipeline=InteractionPipeline.from_dmf_config(config),
        scoring_engine=ScoringEngine.from_dmf_config(config),
        embedding_engine=embedding_engine,
    )


def _ingest(
    runtime: BenchmarkRuntime,
    text: str,
    provenance: InteractionProvenance,
) -> None:
    report, vector = runtime.pipeline.analyze_interaction_with_vector(
        text,
        provenance=provenance,
    )
    if vector is None:
        raise BenchmarkError("DMF pipeline unexpectedly skipped a benchmark turn")
    runtime.scoring_engine.calculate_score(report, text=text)
    runtime.temporal_memory.add_interaction(text, report, vector)


def seed_ltm(runtime: BenchmarkRuntime, dataset: BenchmarkDataset) -> dict[str, object]:
    """Ingest deterministic seed/filler turns and prove every seed reached LTM."""
    for turn_index, turn in enumerate(dataset.seed):
        _ingest(
            runtime,
            turn.text,
            InteractionProvenance(
                role=turn.role,
                source_turn=turn_index,
                is_user_correction=turn.is_user_correction,
                is_preference_update=turn.is_preference_update,
                is_constraint=turn.is_constraint,
            ),
        )

    filler_turns = 0
    missing_seed_ids: list[str] = []
    while filler_turns <= dataset.filler.max_turns:
        archived_texts = {record.text for record in runtime.hook.read_all()}
        missing_seed_ids = [
            turn.seed_id for turn in dataset.seed if turn.text not in archived_texts
        ]
        if not missing_seed_ids:
            break
        if filler_turns == dataset.filler.max_turns:
            break
        filler_turns += 1
        text = dataset.filler.template.format(index=filler_turns)
        _ingest(
            runtime,
            text,
            InteractionProvenance(role="assistant", source_turn=len(dataset.seed) + filler_turns),
        )

    archived_count = runtime.hook.count()
    if archived_count <= 0:
        raise BenchmarkError("Chroma count is zero after bounded seed pressure")
    if missing_seed_ids:
        raise BenchmarkError(
            "Seed evidence did not reach LTM after bounded pressure: "
            + ", ".join(missing_seed_ids)
        )

    recoverability: list[dict[str, object]] = []
    unrecoverable: list[str] = []
    for case in dataset.cases:
        vector = runtime.embedding_engine.get_embedding(case.question)
        hits = runtime.hook.search_raw(vector.tolist(), k=archived_count)
        corpus = "\n".join(hit.record.text for hit in hits)
        score = score_text(corpus, case.evidence_terms)
        recovered = score["coverage"] == 1.0
        recoverability.append(
            {"case_id": case.case_id, "recovered": recovered, "hit_count": len(hits)}
        )
        if not recovered:
            unrecoverable.append(case.case_id)
    if unrecoverable:
        raise BenchmarkError(
            "Archived evidence is not recoverable through Chroma search: "
            + ", ".join(unrecoverable)
        )

    return {
        "seed_turn_count": len(dataset.seed),
        "filler_turn_count": filler_turns,
        "chroma_count_after_seed": archived_count,
        "all_seed_turns_archived": True,
        "direct_search_recoverability": recoverability,
    }


def run_cases(
    runtime: BenchmarkRuntime,
    dataset: BenchmarkDataset,
    ollama: OllamaClient,
) -> tuple[list[dict[str, object]], bool]:
    """Run isolated retrieval/chat cases and return results plus partial flag."""
    results: list[dict[str, object]] = []
    partial_failure = False
    for case in dataset.cases:
        result: dict[str, object] = {
            "id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected_answer": case.expected_answer,
            "model_response": None,
            "dmf_context": "",
            "ollama_latency_ms": None,
            "retrieval_success": False,
            "retrieval_score": None,
            "answer_score": None,
            "error": None,
        }
        try:
            context = runtime.memory.render_context(case.question)
            retrieval_score = score_text(context, case.evidence_terms)
            result.update(
                {
                    "dmf_context": context,
                    "retrieval_success": retrieval_score["coverage"] == 1.0,
                    "retrieval_score": retrieval_score,
                }
            )
            response, latency_ms = ollama.chat(case.question, context)
            answer_score = score_text(response, case.answer_terms, case.obsolete_terms)
            result.update(
                {
                    "model_response": response,
                    "ollama_latency_ms": latency_ms,
                    "answer_score": answer_score,
                }
            )
        except Exception as exc:  # isolate one case and preserve a partial report
            partial_failure = True
            result["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        results.append(result)
    return results, partial_failure


def aggregate_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate completed cases without turning quality into an exit-code gate."""
    completed = [result for result in results if result["error"] is None]
    retrieval_successes = sum(bool(result["retrieval_success"]) for result in completed)
    answer_scores = [
        float(result["answer_score"]["score"])
        for result in completed
        if isinstance(result.get("answer_score"), dict)
    ]
    latencies = [
        float(result["ollama_latency_ms"])
        for result in completed
        if isinstance(result.get("ollama_latency_ms"), (int, float))
    ]
    total = len(results)
    retrieval_rate = retrieval_successes / total if total else 0.0
    mean_answer = sum(answer_scores) / len(answer_scores) if answer_scores else 0.0
    return {
        "case_count": total,
        "completed_case_count": len(completed),
        "failed_case_count": total - len(completed),
        "retrieval_success_count": retrieval_successes,
        "retrieval_success_rate": round(retrieval_rate, 4),
        "mean_answer_score": round(mean_answer, 4),
        "total_ollama_latency_ms": round(sum(latencies), 3),
        "mean_ollama_latency_ms": round(sum(latencies) / len(latencies), 3)
        if latencies
        else None,
        "quality_targets": {
            "retrieval_at_least_8_of_10": retrieval_successes >= 8,
            "mean_answer_score_at_least_0_70": mean_answer >= 0.70,
        },
    }


def read_git_commit(repo_root: Path) -> str | None:
    """Read the local Git commit without executing a subprocess."""
    try:
        git_dir = repo_root / ".git"
        if git_dir.is_file():
            marker = git_dir.read_text(encoding="utf-8").strip()
            git_dir = (repo_root / marker.removeprefix("gitdir: ")).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            loose_ref = git_dir / head[5:]
            if loose_ref.exists():
                return loose_ref.read_text(encoding="utf-8").strip()
            packed_refs = (git_dir / "packed-refs").read_text(encoding="utf-8")
            suffix = f" {head[5:]}"
            return next(
                line.split(" ", 1)[0]
                for line in packed_refs.splitlines()
                if line.endswith(suffix)
            )
        return head
    except (OSError, StopIteration):
        return None


def new_report(dataset: BenchmarkDataset | None, model: str, base_url: str) -> dict[str, object]:
    """Create a sanitized report envelope before infrastructure mutation."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "started",
        "benchmark_id": dataset.benchmark_id if dataset else None,
        "dataset_version": dataset.schema_version if dataset else None,
        "started_at": datetime.now(UTC).isoformat(),
        "dmf_commit": read_git_commit(REPO_ROOT),
        "ollama": {"model": model, "base_url": base_url},
        "chroma": None,
        "scorer": {
            "version": SCORER_VERSION,
            "formula": "max(0, required_group_coverage - 0.5 * promoted_obsolete_ratio)",
        },
        "seed": None,
        "cases": [],
        "aggregate": None,
        "errors": [],
    }


def default_output_path() -> Path:
    """Return a timestamped ignored result path."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "integrationtest" / "results" / f"ltm_benchmark_{stamp}.json"


def write_report(path: Path, report: dict[str, object]) -> None:
    """Persist the JSON report atomically enough for a single local process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse local benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "integrationtest" / "benchmark_cases.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run preflight, seed, retrieval, Ollama Q&A, and report generation."""
    args = parse_args(argv)
    os.chdir(REPO_ROOT)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    dataset: BenchmarkDataset | None = None
    runtime: BenchmarkRuntime | None = None
    raw_base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)
    raw_model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    output_path = args.output or default_output_path()
    report = new_report(None, raw_model.strip(), "unvalidated")
    exit_code = 2

    try:
        dataset = load_dataset(args.dataset.resolve())
        if len(dataset.cases) != MAX_CASES:
            raise BenchmarkError("The versioned benchmark dataset must contain exactly 10 cases")
        base_url = validate_loopback_url(raw_base_url)
        model = validate_model_name(raw_model)
        chroma_host = validate_loopback_host(
            os.getenv("CHROMA_HOST", DEFAULT_CHROMA_HOST), field_name="CHROMA_HOST"
        )
        chroma_port = parse_port(
            os.getenv("CHROMA_PORT", str(DEFAULT_CHROMA_PORT)), field_name="CHROMA_PORT"
        )
        report = new_report(dataset, model, base_url)
        report["chroma"] = {
            "host": chroma_host,
            "port": chroma_port,
            "tenant": "default_tenant",
            "database": "default_database",
        }

        require_local_embedding_cache(REPO_ROOT)
        with OllamaClient(base_url, model) as ollama:
            ollama.preflight()
            collection_name = f"dmf_benchmark_{uuid4().hex}"
            base_config = load_dmf_config(REPO_ROOT / "dmf_settings.toml")
            config = build_benchmark_config(
                base_config,
                collection_name=collection_name,
                chroma_host=chroma_host,
                chroma_port=chroma_port,
            )
            runtime = build_runtime(config)
            report["chroma"]["collection"] = collection_name  # type: ignore[index]
            report["seed"] = seed_ltm(runtime, dataset)
            results, partial_failure = run_cases(runtime, dataset, ollama)
            report["cases"] = results
            report["aggregate"] = aggregate_results(results)
            report["status"] = "partial_failure" if partial_failure else "complete"
            exit_code = 3 if partial_failure else 0
    except (BenchmarkError, httpx.HTTPError, OSError, ValueError) as exc:
        report["status"] = "infrastructure_error"
        report["errors"] = [f"{type(exc).__name__}: {str(exc)[:500]}"]
        exit_code = 2
    except Exception as exc:  # CLI boundary: always emit a diagnostic report
        report["status"] = "infrastructure_error"
        report["errors"] = [f"unexpected {type(exc).__name__}: {str(exc)[:500]}"]
        exit_code = 2
    finally:
        if runtime is not None:
            try:
                runtime.hook.clear()
            except Exception as exc:  # cleanup must not hide the primary result
                report.setdefault("errors", []).append(
                    f"cleanup {type(exc).__name__}: {str(exc)[:300]}"
                )
                if exit_code == 0:
                    report["status"] = "partial_failure"
                    exit_code = 3
        report["finished_at"] = datetime.now(UTC).isoformat()

    try:
        write_report(output_path, report)
    except OSError as exc:
        print(f"Benchmark report write failed: {type(exc).__name__}", file=sys.stderr)
        return 4

    print(f"Benchmark status: {report['status']}")
    print(f"Report: {output_path}")
    if report["errors"]:
        for error in report["errors"]:
            print(f"Error: {error}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
