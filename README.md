# ITSM Insights

Analytics and machine-learning backbone for IT service-management tickets:
understand what users report, measure how tickets flow, predict routing and
sentiment, and catch volume anomalies before they become outages.

Four deliverables, one codebase:

| # | Question | Where |
|---|----------|-------|
| 0 | Can we analyse this at all - what personal data is in the text? | `itsm.privacy` + `itsm privacy` |
| 1 | What families of tickets exist, and what do the status updates say? | `itsm.analysis` + `itsm analyze` |
| 2 | Can we predict the category, the mood, the next spike - and how well? | `itsm.models` + `itsm.pipelines` + `itsm train` / `itsm eval` |
| 3 | How is this served, and shown? | `itsm.api` + `itsm serve`, `itsm.app` + `itsm app` |

---

## Quickstart

```bash
uv sync                      # create .venv and install (Python >= 3.11)
uv run itsm data build       # generate the dataset (offline, seeded)
uv run itsm privacy anonymise  # strip personal data from the free text
uv run itsm analyze designations
uv run itsm analyze lifecycle
uv run itsm train classifier
uv run itsm eval sentiment
uv run itsm eval signals

uv run itsm app              # Streamlit UI   -> http://localhost:8501
uv run itsm serve            # FastAPI + docs -> http://127.0.0.1:8000/docs
```

> First time and in a hurry: `uv sync && uv run itsm app`, then click
> **Prepare the demo** - it builds the dataset and trains the models for you.

Everything above runs **CPU-only and offline**, with scikit-learn defaults.
Transformer backends are opt-in:

```bash
uv sync --extra hf
ITSM_EMBEDDING_BACKEND=embeddings ITSM_SENTIMENT_BACKEND=transformer uv run itsm eval sentiment
```

The Presidio PII analyser is a separate extra - it pulls spaCy, not torch - and
runs standalone so it can be benchmarked against the other two detectors on its
own recognisers:

```bash
uv sync --extra presidio
uv run python -m spacy download en_core_web_lg
uv run itsm privacy eval --backend rules --backend transformer --backend presidio --isolate
```

Poetry users: `poetry install && poetry run itsm --help` (the `pyproject.toml`
is PEP 621, both tools read it).

---

## Data

The project runs on any of four sources, selected by `ITSM_DATASET`:

| Value | What it is |
|-------|-----------|
| `synthetic` *(default)* | Seeded generator: bilingual FR/EN designations, full audit trail, **injected anomalies with a ground-truth manifest**. |
| `uci_incident_log` | [UCI dataset 498 - incident management process enriched event log](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log): 141 712 events / 24 918 real ServiceNow incidents. |
| `huggingface` | [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) and similar multilingual ticket corpora. |
| `csv` | Any ServiceNow / GLPI / Jira export; column aliases in `itsm/data/sources.py`. |

All four are coerced into one canonical schema (`itsm/data/schema.py`), modelled
on the UCI/ServiceNow field names, so **every pipeline runs unchanged on real
data**. Details and trade-offs: [`docs/01-data.md`](docs/01-data.md).

---

## Results at a glance

Measured on the default dataset (7 897 tickets, 12 categories, FR/EN), full
detail and caveats in [`docs/06-results.md`](docs/06-results.md):

| | |
|---|---|
| Category classification | **macro-F1 0.864** vs 0.027 majority baseline; ~90% of volume auto-routable at ~93% accuracy above a 0.8 confidence threshold |
| Ticket families | 36 families auto-selected, mean purity 0.62 - clusters are *finer* than the 12-category taxonomy, which is the actionable finding |
| Sentiment (lexicon) | macro-F1 0.804, kappa 0.664; 87% recall on frustrated requesters at 61% precision |
| Weak signals | all 4 injected anomalies caught, ~1.7 false alarms per 100 series-days, slow ramp flagged 12 days in |
| Anonymisation | **0 of 2 087 personal values survive** in the output (rule engine), F1 0.9995, **zero utility cost** to the classifier |

The dataset injects 7% label noise, 13% ambiguous wordings and 12% typos on
purpose: without them every estimator scores a perfect 1.0 and the evaluation
measures nothing. It also plants personal data with character-level ground
truth, which is what makes the anonymisation measurable rather than merely
demonstrable.

## Layout

```
src/itsm/
├── config.py            # pydantic-settings, the single source of configuration
├── cli.py               # typer CLI, a thin shell over the pipelines
├── data/                # schema, public-dataset adapters, synthetic generator
├── text/                # normalisation and vectorisation (TF-IDF | embeddings)
├── analysis/            # clustering + c-TF-IDF labelling, lifecycle statistics
├── privacy/             # PII detection (rules | HF NER | Presidio), entity-tag redaction, scoring
├── models/              # classifier, sentiment, weak-signal detector, artifact store
├── pipelines/           # train / evaluate / report, one run() each
├── api/                 # FastAPI app, routers, schemas, dependencies
└── app/                 # Streamlit UI: state (caching), charts, one module per page
docs/                    # data, analysis, modelling, API, app, results, ADR
notebooks/               # walkthrough of the analysis; anonymisation backend comparison
scripts/                 # summarize_run.py - headline figures of a full run
tests/                   # hermetic suite on a small generated dataset
```

The CLI, the API and the Streamlit app are three clients of the same library.
None of them contains analysis logic, so they cannot drift apart.

## Documentation

* [`docs/01-data.md`](docs/01-data.md) - datasets, canonical schema, what the generator simulates.
* [`docs/02-analysis.md`](docs/02-analysis.md) - clustering method, cluster naming, lifecycle indicators.
* [`docs/03-modeling.md`](docs/03-modeling.md) - the three model families, evaluation protocols, how to read the numbers.
* [`docs/04-api.md`](docs/04-api.md) - endpoints, serving model, what is deliberately not included.
* [`docs/05-app.md`](docs/05-app.md) - Streamlit architecture, caching strategy, the six pages.
* [`docs/06-results.md`](docs/06-results.md) - **what the models actually score**, and what those numbers do and do not mean.
* [`docs/07-privacy.md`](docs/07-privacy.md) - **anonymisation**: what personal data is in the text, the three detectors, and how the result is measured.
* [`docs/adr/0001-stack-and-baselines.md`](docs/adr/0001-stack-and-baselines.md) - why these choices.
* [`docs/adr/0002-anonymisation.md`](docs/adr/0002-anonymisation.md) - why anonymisation is a pipeline stage with interchangeable detectors.

## API in one glance

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/healthz`, `/readyz` | Liveness; readiness reports which artifacts are missing. |
| `GET` | `/v1/models` | Trained artifacts **with their evaluation metrics**. |
| `POST` | `/v1/classify` | Category prediction with confidence and an abstention flag. |
| `POST` | `/v1/sentiment` | Frustration polarity with the matched evidence. |
| `GET` | `/v1/clusters`, `POST /v1/clusters/assign` | Discovered families; assign a new designation. |
| `GET` | `/v1/stats/overview`, `/v1/stats/by/{dim}`, `/v1/stats/flow` | Volume, MTTR, SLA, backlog, transitions. |
| `GET` | `/v1/tickets/{id}/trace` | One ticket and its full status history. |
| `POST` | `/v1/signals/scan`, `GET /v1/signals/series` | Spikes, emerging weak signals, and the evidence behind them. |
| `POST` | `/v1/anonymise` | Remove personal data from free text before it goes anywhere else. |

## Development

```bash
uv run pytest             # hermetic, runs in a temporary workspace
uv run ruff check src tests
uv run ruff format src tests
uv run mypy
```
