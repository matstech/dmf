# DMF (Deterministic Memory Framework)

Welcome to the documentation for DMF (Deterministic Memory Framework).

## Introduction
*(TODO: Add a detailed description of what the framework is, how the Deterministic Memory Framework operates conceptually, and its core design principles.)*

## Quickstart & Usage

The main entry point for querying the active memory is the `Memory` class. To initialize the framework, you typically provide a configuration, an embedding engine, and a temporal memory instance (which wraps your chosen Long-Term Memory backend, like ChromaDB).

Here is an example of how to configure and use these core APIs:

```python
from dmf.utils.config_loader import DMFConfig
from dmf.analysis import EmbeddingEngine, NLPEngine
from dmf.memory import ChromaLTMHook, TemporalMemory, Memory

# 1. Load the configuration
config = DMFConfig.load("dmf_settings.toml")

# 2. Initialize engines and LTM hook
embedding_engine = EmbeddingEngine(model_name=config.analysis.embedding_model)
nlp_engine = NLPEngine()
ltm_hook = ChromaLTMHook(persist_directory="./data/chroma")

# 3. Initialize TemporalMemory
temporal_memory = TemporalMemory(
    ltm_hook=ltm_hook,
    nlp_engine=nlp_engine,
    time_decay_halflife=config.temporal.time_decay_halflife,
    recency_window_size=config.temporal.recency_window_size
)

# 4. Initialize the Memory facade
memory = Memory.from_dmf_config(
    config=config,
    temporal_memory=temporal_memory,
    embedding_engine=embedding_engine
)

# --- Usage ---

# Retrieve structured evidence
evidence = memory.retrieve("What did we discuss about the new architecture?")

# Or render a prompt-ready context string directly
context_string = memory.render_context("What did we discuss about the new architecture?")
print(context_string)
```
