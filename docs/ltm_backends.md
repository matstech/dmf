# Long-term memory backends

DMF separates the `LTMHook` contract from its concrete storage backends. The
backend is normally selected through `[ltm]` in `dmf_settings.toml`; direct
construction remains available for custom applications and tests.

## Backend overview

| Backend | Retrieval | Persistence | Intended use |
|---|---|---|---|
| `FileLTMHook` | No semantic search | Append-only JSONL | Development and audit trails |
| Chroma embedded | Semantic search | Local Chroma directory | Single-process applications |
| Chroma server | Semantic search | Managed by the Chroma service | Multiple clients and service deployments |
| `NullLTMHook` | No | No | Disabled LTM and isolated tests |

Chroma embedded remains the default connection mode. Existing configurations
that do not define `chroma_mode` continue to use `PersistentClient` and
`chroma_path`.

## Canonical imports

```python
from dmf.memory.ltm_hooks import ChromaLTMHook, FileLTMHook
from dmf.memory.ltm_hooks.chroma_client import (
    ChromaConnectionConfig,
    ChromaConnectionMode,
)
```

The same hook classes remain re-exported from `dmf` and `dmf.memory`.

## Embedded Chroma

Use embedded mode when DMF owns the local Chroma data directory.

```toml
[ltm]
enabled = true
storage_type = "chroma"
chroma_mode = "embedded"
chroma_path = "data/ltm_chroma"
collection_name = "dmf_memory"
```

Direct construction remains backward compatible:

```python
from dmf.memory import ChromaLTMHook

hook = ChromaLTMHook(persist_directory="data/ltm_chroma")
```

The directory is created automatically. Network retry is not installed because
embedded mode does not use the HTTP client.

## Chroma server

Server mode contacts a separately managed Chroma service and does not create
`chroma_path`.

```toml
[ltm]
enabled = true
storage_type = "chroma"
chroma_mode = "server"
chroma_host = "localhost"
chroma_port = 8000
chroma_ssl = false
chroma_tenant = "default_tenant"
chroma_database = "default_database"
chroma_auth_token_env = ""
collection_name = "dmf_memory"
```

Direct construction uses an explicit connection object:

```python
from dmf.memory import ChromaLTMHook
from dmf.memory.ltm_hooks.chroma_client import (
    ChromaConnectionConfig,
    ChromaConnectionMode,
)

connection = ChromaConnectionConfig(
    mode=ChromaConnectionMode.SERVER,
    host="localhost",
    port=8000,
)
hook = ChromaLTMHook(connection=connection)
```

An already constructed Chroma client can be passed with `client=...`. Injected
clients are used unchanged, so their connection lifecycle and resilience remain
the caller's responsibility.

## Authentication

The TOML contains only the name of the environment variable holding the token:

```toml
chroma_auth_token_env = "DMF_CHROMA_TOKEN"
```

```bash
export DMF_CHROMA_TOKEN="your-server-token"
```

DMF reads the variable only when LTM is enabled with the Chroma server backend.
If a variable name is configured but its value is missing, empty, or
whitespace-only, initialization fails before the client is constructed. The
token is sent as a Bearer token and is excluded from configuration
representations, logs, and benchmark reports.

Leave `chroma_auth_token_env = ""` for an unauthenticated local service.

## Retry behavior

The server client applies retry at the HTTP layer, including requests made
during client bootstrap.

Retries are limited to:

- `httpx.TransportError`, including connection and timeout failures;
- HTTP 502, 503, and 504 responses.

The policy uses exponential backoff with full jitter and at most five total
attempts. Client errors, authentication failures, validation errors, and other
HTTP responses are not retried. Once the retry budget is exhausted, the error
is propagated through Chroma's normal error mapping.

Retry timing is intentionally fixed in this release and is not configurable in
TOML.

## Memory cards and local files

Raw LTM records remain canonical. When cards are enabled, Chroma uses a
separate card collection. The optional JSONL card audit file remains local even
when Chroma runs in server mode; set `cards_path` explicitly when the default
path is not appropriate for the client host.

## Local Chroma service

The repository includes `compose.chroma.yml` with a version-pinned Chroma
0.6.3 service and a persistent named volume.

```bash
make chroma-up
make test-integration
make chroma-down
```

Stopping the service does not delete its volume. To use a different host port,
pass the same value to Compose and the test:

```bash
CHROMA_PORT=18000 make chroma-up
CHROMA_PORT=18000 make test-integration
CHROMA_PORT=18000 make chroma-down
```

The standard `make test` command excludes integration tests and does not
require Docker.

## Local Ollama functional benchmark

The optional benchmark exercises the complete local path: DMF ingestion,
archival to Chroma server, retrieval, context rendering, and an isolated Ollama
agent. It is a functional diagnostic and not a model-quality CI gate.

Prerequisites:

- the Chroma Compose service is running;
- Ollama is listening on a loopback address;
- `qwen2.5:0.5b`, or the model selected by `OLLAMA_MODEL`, is already installed;
- the configured embedding assets are already available locally.

```bash
make chroma-up
ollama serve  # omit when Ollama is already running
ollama list
OLLAMA_BASE_URL=http://localhost:11434 \
OLLAMA_MODEL=qwen2.5:0.5b \
make benchmark-ltm-local
make chroma-down
```

The runner never downloads models or images. It accepts only loopback Ollama
and Chroma endpoints and writes sanitized reports under the ignored
`integrationtest/results/` directory.

## Version compatibility

The Python dependency supports Chroma `>=0.6.3,<0.7.0`, and the local Compose
image is pinned to `0.6.3`. The retry integration depends on Chroma 0.6.x HTTP
internals and must be reviewed before upgrading to Chroma 0.7 or later.

## Migrating legacy imports

The old modules remain compatibility shims:

```python
# Deprecated
from dmf.memory.chroma_ltm import ChromaLTMHook
from dmf.memory.ltm_engine import FileLTMHook
```

They still return the canonical classes but emit `DeprecationWarning`. Migrate
to one of the supported public forms:

```python
from dmf import ChromaLTMHook, FileLTMHook
# or
from dmf.memory import ChromaLTMHook, FileLTMHook
# or
from dmf.memory.ltm_hooks import ChromaLTMHook, FileLTMHook
```

## Troubleshooting

### The server path is not created

This is expected in server mode. `chroma_path` belongs to embedded mode; the
server owns its persistence.

### Authentication works locally but fails in another environment

Confirm that `chroma_auth_token_env` names an environment variable available
to the DMF process. Do not place the token value directly in TOML.

### Startup takes longer while the server is unavailable

Client bootstrap is covered by the bounded retry policy. Inspect warning logs
for the attempt number and delay; after five attempts the original failure is
propagated.

### The integration test is not collected by `make test`

This is intentional. Run `make test-integration` after starting the Compose
service.
