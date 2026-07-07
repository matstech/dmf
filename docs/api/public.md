# Public API

## Memory facade

::: dmf.memory.api.Memory
    options:
      filters:
        - "!^_[^_]"
        - "!^__init__$"

## Temporal memory

::: dmf.memory.temporal_memory.TemporalMemory
    options:
      filters:
        - "!^_[^_]"

## Chroma LTM hook

::: dmf.memory.ltm_hooks.chroma_hook.ChromaLTMHook
    options:
      filters:
        - "!^_[^_]"

## File LTM hook

::: dmf.memory.ltm_hooks.file_hook.FileLTMHook
    options:
      filters:
        - "!^_[^_]"

## Chroma connection mode

::: dmf.memory.ltm_hooks.chroma_client.ChromaConnectionMode

## Chroma connection configuration

::: dmf.memory.ltm_hooks.chroma_client.ChromaConnectionConfig

## Chroma client factory

::: dmf.memory.ltm_hooks.chroma_client.build_chroma_client
