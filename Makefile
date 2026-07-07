.PHONY: lock install build test test-integration benchmark-ltm-local chroma-up chroma-down check compile docs-serve docs-build

# Aggiorna poetry.lock 
lock:
	poetry lock 

# Installa le dipendenze (incluse quelle dev) e il modello spaCy
install:
	poetry install --with dev --no-interaction
	poetry run python -m spacy download en_core_web_sm

# Verifica i metadati del pacchetto
check:
	poetry check

# Compila i sorgenti Python (controllo sintattico)
compile:
	poetry run python -m compileall dmf tests integrationtest

# Esegue la suite di test
test:
	poetry run pytest

# Avvia Chroma 0.6.3 e attende che il servizio sia pronto
chroma-up:
	docker compose -f compose.chroma.yml up -d --wait

# Esegue solo i test che richiedono il server Chroma locale
test-integration:
	poetry run pytest -o addopts="-v --tb=short" -m integration integrationtest

# Esegue il mini-benchmark locale DMF LTM + Ollama (separato da pytest)
benchmark-ltm-local:
	poetry run python -m integrationtest.run_ltm_benchmark

# Arresta i container senza eliminare il volume persistente
chroma-down:
	docker compose -f compose.chroma.yml down

# Produce il wheel nella cartella dist/
build:
	poetry build --format wheel

# Avvia il server di sviluppo per la documentazione
docs-serve:
	poetry run mkdocs serve

# Costruisce la documentazione statica
docs-build:
	poetry run mkdocs build
