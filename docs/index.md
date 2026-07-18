<p align="center">
  <img src="assets/logo.svg" alt="DMF Logo" width="65"/>
</p>

# Deterministic Memory Framework

DMF provides a deterministic memory lifecycle for conversational applications:
interaction analysis, active-memory pruning, long-term archival, semantic
retrieval, reranking, and prompt-ready context rendering.

## Quickstart

The recommended entry point is the project configuration. It selects the LTM
backend and keeps application wiring independent from whether Chroma runs
embedded or as a server, and whether Qdrant runs in memory or as a server.

```python
from dmf.analysis import EmbeddingEngine, ScoringEngine
from dmf.memory import Memory, TemporalMemory
from dmf.runtime.pipeline import InteractionPipeline
from dmf.utils.config import VectorConfig
from dmf.utils.config_loader import load_dmf_config

config = load_dmf_config("dmf_settings.toml")
pipeline = InteractionPipeline.from_dmf_config(config)
scoring = ScoringEngine.from_dmf_config(config)
temporal_memory = TemporalMemory.from_dmf_config(config)

embedding_engine = EmbeddingEngine(
    VectorConfig(
        model_name=config.nlp.model_name,
        vector_dim=config.nlp.vector_dim,
        window_size=config.capacity.window_size,
    )
)
memory = Memory.from_dmf_config(config, temporal_memory, embedding_engine)

text = "We decided to use a separate service for the vector database."
report, vector = pipeline.analyze_interaction_with_vector(text)
if vector is not None:
    scoring.calculate_score(report, text=text)
    temporal_memory.add_interaction(text, report, vector)

context = memory.render_context("What did we decide about the architecture?")
print(context)
```

`chroma_mode = "embedded"` is the backward-compatible default. Switching to
`chroma_mode = "server"` changes the connection strategy without changing the
application code above. Selecting `storage_type = "qdrant"` with
`qdrant_mode = "memory"` uses volatile Qdrant Local Mode and also keeps the
application code unchanged. Switching to `qdrant_mode = "server"` connects the
same hook to a separately managed Qdrant service.

## Long-term memory backends

DMF includes:

- `FileLTMHook`, an append-only JSONL archival backend;
- `ChromaLTMHook` in embedded mode, with local persistent storage;
- `ChromaLTMHook` in server mode, using a separately managed Chroma service;
- `QdrantLTMHook` in Local Mode, using volatile in-memory Qdrant;
- `QdrantLTMHook` in server mode, using service-managed persistence;
- `NullLTMHook`, for disabled persistence and isolated tests.

See [LTM Backends](ltm_backends.md) for deployment modes, authentication,
retry behavior, migration from legacy imports, Docker integration, and the
local Ollama benchmark.

## Next steps

- [Configuration reference](configuration.md)
- [LTM backends and server deployment](ltm_backends.md)
- [Context rendering](prompt_structure.md)
- [Public API](api/public.md)
- [Internal API](api/internals.md)
