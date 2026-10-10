# Ticket AI

A minimal Streamlit app that shows what an LLM and embeddings add to a ticket extract.
It reads the extract **plus the columns the LLM inferred** (`IncidentType`, `Tone`, `RequestType`,
`Scope`, `RootCause`, `ResolutionSummary`, `AffectedSystem`) and offers three pages:

| Page | What you do | What the AI does |
|---|---|---|
| **New ticket** | Fill a form like a real ticket and submit it | The LLM extracts structured fields, a k-NN returns the most similar past tickets, rules raise alerts |
| **Batch report** | Pick a period and some types, services, groups... | Analyses the batch and renders a report (**HTML**, **PDF**) with the trend and the alerts |
| **Dashboard** | Choose a granularity and a breakdown | Charts volume and distribution over time, with a **projection** and its uncertainty band |

It runs **offline out of the box**: with no API key it generates a synthetic extract and replaces the LLM
with a local keyword heuristic (clearly labelled in the UI), so you can try everything before connecting anything.

## Quick start

```bash
cd ticket_ai
pip install -r requirements.txt
streamlit run frontend/app.py
```

Optional, for better results:

```bash
pip install sentence-transformers   # multilingual embeddings (otherwise TF-IDF is used)
pip install weasyprint              # PDF export (needs the Pango library on the system)
```

Run the tests with `pip install -r requirements-dev.txt && pytest` (42 tests, offline, about 5 seconds).

## Architecture

```
ticket_ai/
├── backend/               computation only: no Streamlit import anywhere
│   ├── schema.py          column names, allowed values, header normalisation
│   ├── config.py          settings read from environment variables / .env
│   ├── data.py            TicketStore: load the extract, filter it
│   ├── mock_data.py       synthetic extract (same columns as the real one)
│   ├── privacy.py         masks emails, phones, IBANs, cards before an API call
│   ├── llm.py             LLM client, offline fallback, ticket enrichment, batch summary
│   ├── similarity.py      embeddings + k-NN over past tickets
│   ├── alerts.py          ticket-level alert rules
│   ├── analytics.py       volumes, KPIs, projection, growth, batch-level alerts
│   └── report.py          BatchReport -> HTML / PDF
├── frontend/              rendering only: no business logic
│   ├── app.py             entry point, navigation, sidebar
│   ├── state.py           cached resources (store, index, LLM)
│   ├── components.py      alert boxes, HTML preview
│   ├── charts.py          Plotly figures
│   └── views/             one module per page
└── tests/
```

The rule that keeps it simple: **`frontend` asks, `backend` answers.** A page collects inputs, calls a
backend function and draws the result. Every number on screen and in the report comes from the same
functions, so they cannot disagree, and the backend can be tested without a browser.

## Your own data

```bash
TICKETS_PATH=/path/to/extract.csv streamlit run frontend/app.py
```

CSV (`;` or `,`, UTF-8 or Windows-1252) and XLSX are supported. Headers are matched **ignoring accents,
case and spacing**, so `Temps écoulé` and `Temps ecoulé` both work, as does the correctly spelled
`Commentaire additionnel`. Dates are read day-first and numbers accept decimal commas.

Needed at minimum: `Numéro`, `Ouvert le`, `Description courte`, `Description`. Any other missing column is added
empty. The LLM-inferred columns are what the report and dashboard group by: if they are empty, the sidebar says so.
`TGI` is treated as an opaque category.

## The LLM

Any **OpenAI-compatible** chat API works (OpenAI, an enterprise gateway, vLLM, Ollama...):

```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=...
```

- **Ticket form**: the model returns six fields as JSON. Each is validated against the allowed values; a wrong
  or missing answer falls back to a safe default instead of breaking the page.
- **If the call fails** (network, quota, bad JSON), the local heuristic is used and the page says so. Nothing crashes.
- **What leaves the machine**: the ticket text with emails, phone numbers, IBANs and card numbers masked
  (`LLM_REDACT_PII=false` to disable). The **report summary** sends only aggregated figures, never ticket text.
- **Another provider**: `LLMClient.complete` in `backend/llm.py` is the only place that knows about HTTP.

> **Limit:** masking uses regular expressions, so it catches data with a *format* but **not names**. A ticket saying
> "contact Camille Dubois" still carries the name to the API. For real data, swap `redact_pii` for a proper
> detector (Presidio, GLiNER); its signature is the only contract.

## How it works

### Similar tickets

```
similarity = 0.7 x cosine(text embeddings) + 0.3 x share of matching categories
```

Text is *short description + description*. Categories are service, assignment group, incident type,
affected system and site, and include what the LLM inferred for the new ticket. Both terms are in `[0, 1]`, so a
score reads directly: 0.82 means close wording and mostly the same context. The search is a brute-force dot
product (milliseconds up to hundreds of thousands of tickets), so there is no vector database to run. Ties go to the most
recent ticket. Embeddings are cached on disk after the first run.

### Alerts

Alerts are **plain rules** over the LLM's facts and the history, so each one can be read, tested and tuned (`Thresholds` in
`backend/alerts.py`).

| Alert | Raised when |
|---|---|
| Possible major incident | Urgency **and** impact are High |
| Security-related ticket | The LLM classes it as `Sécurité` |
| Wide impact | Scope is `Site` or `Multi-sites` |
| Escalation risk | Tone is frustrated, urgent or aggressive |
| Emerging pattern | 5+ similar tickets (similarity above 0.6) in the last 14 days; critical at 10+ |
| Similar tickets were hard to resolve | Among the 10 closest: 40%+ missed SLA, or 0.5+ reopenings on average |
| Personal data detected | Emails, phones, IBANs or cards found in the text |

The batch report adds **volume spike**, **sustained rise**, **growing topic** (type, system or root cause) and
**SLA / tone drift**. A topic alert needs the rise to be large *and* statistically unlikely to be noise (Poisson test).

### Report

`build_report` returns a `BatchReport`; `to_html()` gives one self-contained file and `to_pdf()` runs that same HTML
through WeasyPrint, so the two cannot differ. Charts are **static SVG** (matplotlib), not Plotly: a report has to open
offline and print. Every dynamic value is HTML-escaped, which matters because the preview is an iframe.

### Projection

Damped Holt exponential smoothing on weekly or monthly counts: a level plus a trend that is damped so a recent
rise does not extrapolate forever. The shaded band is the usual approximate 95% prediction interval and widens with
the horizon. Partial first and last periods are excluded so they do not distort the trend.

> **Limit:** it ignores seasonality on purpose. With one or two years of weekly data there is too little to estimate
> it, and a model claiming more than it knows is worse than a plain one. Read the band, not just the line.

## Notes

- **Theme**: `.streamlit/config.toml` is the same Thales theme as the main ITSM app. Streamlit reads it from the
  folder you launch from, so run from `ticket_ai/` (or from the repo root, whose config has the same theme).
- **Synthetic data** plants two events so there is something to find: a VPN surge after a client update (last 3 weeks)
  and a slow Outlook ramp (last 8 weeks). The alerts and the projection should surface both.
- **PDF on Windows**: WeasyPrint needs GTK/Pango. If it is missing, the button is disabled and the HTML report still works
  (print it to PDF from the browser).
- Tested with Streamlit 1.65 and Python 3.12.
