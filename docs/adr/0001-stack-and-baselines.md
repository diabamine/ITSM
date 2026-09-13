# ADR 0001 - Stack, baselines and evaluation posture

* **Status**: accepted
* **Date**: 2026-09-11
* **Context**: initial build of the ITSM analytics project

## Context

Build, from scratch, an analysis + modelling + serving project over ITSM
tickets, with the constraints: Python, uv or Poetry, FastAPI, minimal,
structured, documented, open-source models allowed. No real ticket extract is
available at build time, and the target environment may be offline.

## Decisions

### 1. uv over Poetry

Both read the same PEP 621 `pyproject.toml`, so the choice is reversible and
Poetry users are unaffected (`poetry install` works). uv is chosen for install
speed and for producing a lockfile plus a virtualenv in one command, which
matters in CI and in restricted environments.

### 2. A synthetic generator as the default dataset, with real-dataset adapters

No public corpus has free text, a taxonomy and an audit trail at once
(see [`01-data.md`](../01-data.md)). Rather than pick one and leave two thirds
of the project unimplemented, the canonical schema is modelled on the UCI
ServiceNow event log and a seeded generator fills it.

*Consequence*: the project runs offline on day one, and - critically - the
weak-signal detector and the sentiment model become **evaluable**, because the
generator knows when it injected an anomaly and which mood produced which
wording. Adapters for UCI 498, Hugging Face corpora and arbitrary CSV exports
mean the same pipelines run on real data unchanged.

*Risk accepted*: scores obtained on generated data are an upper bound. Real
designations are messier, and the taxonomy overlap is worse. The mitigation is
that the evaluation *protocol* - temporal split, macro-F1, coverage curve,
detection delay - transfers even when the numbers do not.

### 3. Classical NLP as the default, transformers as an opt-in

TF-IDF (word + char n-grams) with a calibrated linear model is the default for
classification and clustering; sentence-transformers and a multilingual
sentiment model are behind the `[hf]` extra.

*Rationale*: short, templated, domain-specific texts are the regime where
linear models are closest to transformers; they train in seconds on CPU, run
offline, and are inspectable - which matters more than two points of F1 when a
support team has to trust the routing. The alternative backend is wired into
the same interface and scored on the same split, so the trade-off can be
re-measured rather than re-argued.

*Consequence*: `[hf]` pulls torch (~2 GB). Keeping it optional means a clone
installs in seconds.

### 4. Robust, seasonality-aware counting statistics over a forecasting model

For weak signals, a median/MAD baseline on same-weekday history with a Poisson
tail test and an EWMA of residuals, rather than Prophet/ARIMA/an autoencoder.

*Rationale*: one year of history does not identify a yearly seasonal term; the
series are small counts, not Gaussians; and a contaminated window is the normal
case, not the exception. Robust statistics degrade gracefully and every alert
comes with an interpretable "expected vs observed" that an on-call engineer can
argue with.

*Consequence*: level shifts are eventually absorbed into the baseline (by
design - they become the new normal). Detecting them permanently requires the
changepoint layer listed in [`03-modeling.md`](../03-modeling.md).

### 5. Evaluation is part of every training run

No pipeline writes a model without writing its metrics next to it, and the API
serves those metrics. Baselines (`majority`, Naive Bayes) are always included.

*Rationale*: the failure mode of this kind of project is not a bad model, it is
an unexamined one.

### 6. Three surfaces, one library

CLI, FastAPI and Streamlit all call `itsm.pipelines` / `itsm.analysis` /
`itsm.models`. None of them contains analysis logic.

*Consequence*: a behaviour change lands in all three at once, and the Streamlit
app is a real client of the same code, not a parallel prototype.

## Alternatives considered

| Alternative | Rejected because |
|-------------|------------------|
| Fine-tuning a small multilingual transformer for classification | Cost and opacity are not justified before a linear baseline has been measured on real data; the hook is in place to do it later. |
| BERTopic for the family discovery | Brings UMAP + HDBSCAN + a transformer as hard dependencies for an output this project reproduces with SVD + KMeans + c-TF-IDF, the same labelling idea, and a `predict` the API can call. |
| An LLM for classification and sentiment | Per-ticket cost and latency at service-desk volume, no offline path, and no stable baseline to evaluate against. The defensible use is assisting *annotation* and drafting resolution summaries - out of scope here. |
| A database-backed API | Premature: the data source is not fixed. The two dataframe dependencies are the seam where it plugs in. |
