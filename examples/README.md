# Examples

These examples demonstrate how to integrate the Deterministic Memory Framework (_dmf_) with generic LLM agents using `litellm`.

## How to run

To avoid polluting the main Poetry environment of the project (since `litellm` is not a core dependency of _dmf_), we recommend running these scripts in an isolated, temporary virtual environment, or by using modern tools like `uv`.

### Option 1: Standard Virtual Environment

```bash
# 1. Enter the examples directory
cd examples

# 2. Create and activate an isolated virtual environment
python -m venv .venv-examples
source .venv-examples/bin/activate

# 3. Install the requirements and the local *dmf* package in editable mode
pip install -r requirements.txt
pip install -e ..

# 4. Export your API key (required for LiteLLM if using OpenAI models)
export OPENAI_API_KEY="sk-..."

# 5. Run the desired example
python litellm_chroma/agent.py
# or
python litellm_file/agent.py
# or the specialized coding agent
python litellm_coder/agent.py
```

### Option 2: Using `uv` (Fast)

If you have [uv](https://docs.astral.sh/uv/) installed, you can run the scripts instantly without manually creating a venv:

```bash
export OPENAI_API_KEY="sk-..."

# Run the script injecting litellm and the current directory (*dmf*) as dependencies:
uv run --with litellm --with .. python litellm_chroma/agent.py
```

## Chroma server mode

The Chroma example defaults to embedded persistence. To use the version-pinned
local server, start it from the repository root with `make chroma-up`, then set
these values in `litellm_chroma/dmf_settings.toml`:

```toml
chroma_mode = "server"
chroma_host = "localhost"
chroma_port = 8000
chroma_ssl = false
chroma_tenant = "default_tenant"
chroma_database = "default_database"
chroma_auth_token_env = ""
```

Use `make chroma-down` from the repository root when finished. It preserves the
Compose data volume. For an authenticated external server, set
`chroma_auth_token_env` to an environment-variable name and export the token in
that variable rather than writing the token into TOML.
