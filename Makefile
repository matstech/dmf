.PHONY: lock install build test check compile docs-serve docs-build

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
	poetry run python -m compileall dmf tests

# Esegue la suite di test
test:
	poetry run pytest

# Produce il wheel nella cartella dist/
build:
	poetry build --format wheel

# Avvia il server di sviluppo per la documentazione
docs-serve:
	poetry run mkdocs serve

# Costruisce la documentazione statica
docs-build:
	poetry run mkdocs build
