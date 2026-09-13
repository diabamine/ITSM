# Convenience targets. Everything works the same with `uv run <cmd>` directly.
.PHONY: install demo data analyze train eval app serve test lint format typecheck check clean

install:            ## create the venv and install the project
	uv sync

demo: data analyze train eval  ## full pipeline, from empty checkout to reports

data:
	uv run itsm data build
	uv run itsm data info

analyze:
	uv run itsm analyze designations
	uv run itsm analyze lifecycle

train:
	uv run itsm train classifier

eval:
	uv run itsm eval sentiment
	uv run itsm eval signals

app:                ## streamlit UI on :8501
	uv run itsm app

serve:              ## fastapi on :8000
	uv run itsm serve

test:
	uv run pytest

lint:
	uv run ruff check src tests

format:
	uv run ruff format src tests

typecheck:
	uv run mypy

check: lint typecheck test

clean:
	rm -rf data artifacts reports .pytest_cache .ruff_cache .mypy_cache
