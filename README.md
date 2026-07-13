<p align="center">
  <img src="docs/assets/logo.svg" alt="DMF Logo" width="65"/>
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2606.03463-b31b1b.svg)](https://arxiv.org/abs/2606.03463)

# DMF

The **Deterministic Memory Framework (DMF)** is a memory management system designed for conversational agents. The idea behind this framework is to avoid unpredictable retention, hallucination, and information loss in memory systems due to an "LLM-in-the-flow" architecture. For this reason, DMF operates on deterministic rules, mathematical temporal decay, and structured semantic retrieval.

At its core, the framework orchestrates _active short-term_ and _long-term memory_ by bridging interaction pipelines with vector databases.

## Motivations

Current approaches to memory in conversational agents rely heavily on the LLM itself to decide what to remember, summarize, or forget. This introduces several fundamental problems:

- **Non-reproducibility:** The same conversation can produce different memory states across runs, making debugging and testing nearly impossible (caching is practically infeasible).
- **Silent information loss:** LLM-driven summarization silently drops details that may be critical later, with no audit trail or recovery mechanism.
- **Hallucinated recall:** When asked to retrieve past context, models may confabulate facts that were never part of the conversation.
- **Opaque retention logic:** There is no way to inspect, tune, or predict which information will survive and which will be discarded.

DMF addresses these issues by replacing probabilistic memory management with a fully deterministic, mathematically grounded pipeline. Every retention decision is traceable to explicit scoring functions, configurable thresholds, and transparent decay curves.

## Core Design Principles

- **Deterministic NLP Analysis:** Employs rule-based parsing to deterministically extract interaction signals, topics, and metrics like information density.
- **Temporal Dynamics:** Automatically manages context size using mathematical time-decay functions and recency windows, naturally prioritizing fresh and highly relevant information.
- **Pluggable Long-Term Memory:** Seamlessly archives interactions into vector databases and triggers context-aware semantic recalls when relevant historical data is needed.
- **Structured Retrieval:** Guarantees that the context injected into your agent's prompt is always optimized through multi-stage candidate generation, answerability reranking, and evidence assembly.

## Installation

You can install it via pip:

```bash
pip install dmf-memory
```

Install the optional Qdrant backend when you want to use local in-memory
Qdrant LTM:

```bash
pip install 'dmf-memory[qdrant]'
```

## Configuration

DMF is fully configurable via a [TOML](https://toml.io/en/) file. You can adjust NLP models, temporal decay rates, and pruning priorities to suit your agent's needs.

Minimal Qdrant Local Mode configuration:

```toml
[ltm]
enabled = true
storage_type = "qdrant"
qdrant_mode = "memory"
```

Qdrant Local Mode uses `QdrantClient(":memory:")`: each client has separate
state, and all archived data disappears when the process exits. Persistent
local Qdrant and Qdrant server connections are outside the current release
scope.

For a comprehensive guide on all configuration parameters, please check our [configuration documentation](configuration.md) in MkDocs.

## Development & Makefile

The project provides a `Makefile` to simplify common development tasks:

- `make install`: Install project dependencies using Poetry.
- `make test`: Run the unit suite without external services.
- `make chroma-up`: Start the version-pinned local Chroma server and wait for readiness.
- `make test-integration`: Run the opt-in Chroma server roundtrip test.
- `make benchmark-ltm-local`: Run the separate 10-case DMF LTM benchmark against local Ollama.
- `make chroma-down`: Stop the local Chroma server while preserving its data volume.
- `make check`: Verify package metadata.
- `make build`: Build the distributable wheel.
- `make docs-serve`: Serve the documentation locally.
- `make docs-build`: Build the static documentation site.

The local server uses `chromadb/chroma:0.6.3`, matching the supported 0.6.x
Python client. See the [configuration documentation](docs/configuration.md#local-chroma-integration-test)
for server mode, authentication, port overrides, and the complete workflow.
The benchmark defaults to the already-installed Ollama model `qwen2.5:0.5b`;
override it with `OLLAMA_MODEL` and keep `OLLAMA_BASE_URL` on a loopback host.
Reports are written under `integrationtest/results/` and are not part of pytest.

## Benchmarks

This framework has been tested and compared with the LoCoMo and LongMemEval frameworks; full details are available in the [dedicated repository](https://github.com/matstech/dmf-benchmarks)


## Citation

If you use the Deterministic Memory Framework in your research, please cite our paper:

```bibtex
@misc{stabile2026dmf,
  title = {DMF: A Deterministic Memory Framework for Conversational AI Agents},
  author = {Matteo Stabile and Enrico Zimuel},
  year = {2026},
  eprint = {2606.03463},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2606.03463}
}
```

## Copyright

Copyright 2026 by [Matteo Stabile](https://www.linkedin.com/in/matteo-stabile) and [Enrico Zimuel](https://www.zimuel.it/).
