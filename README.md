<p align="center">
  <img src="docs/assets/logo.svg" alt="DMF Logo" width="65"/>
</p>

# dmf

The **deterministic memory framework (dmf)** is a memory management system designed for conversational agents. The idea behind this framework is to avoid unpredictable retention, hallucination, and information loss in memory systems due to an "LLM-in-the-flow" architecture. For this reason, DMF operates on deterministic rules, mathematical temporal decay, and structured semantic retrieval.

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

## Configuration

DMF is fully configurable via a [TOML](https://toml.io/en/) file. You can adjust NLP models, temporal decay rates, and pruning priorities to suit your agent's needs.

For a comprehensive guide on all configuration parameters, please check our [configuration documentation](configuration.md) in MkDocs.

## Development & Makefile

The project provides a `Makefile` to simplify common development tasks:

- `make install`: Install project dependencies using Poetry.
- `make test`: Run the test suite.
- `make check`: Verify package metadata.
- `make build`: Build the distributable wheel.
- `make docs-serve`: Serve the documentation locally.
- `make docs-build`: Build the static documentation site.

<!-- ## Citation

If you use the Deterministic Memory Framework in your research, please cite our paper:

```bibtex
@article{,
  title={},
  author={},
  journal={arXiv preprint},
  year={2026}
}
``` -->
