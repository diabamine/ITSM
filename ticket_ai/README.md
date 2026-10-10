# Ticket AI

A minimal Streamlit app on a ticket extract that already contains the columns inferred by an LLM
(`IncidentType`, `Tone`, `RequestType`, `Scope`, `RootCause`, `ResolutionSummary`, `AffectedSystem`).

| Page | What it does |
|---|---|
| **New ticket** | A form like a real ticket. On submit, *your* LLM extracts fields, the closest past tickets are retrieved (k-NN on embeddings + categories) and alerts are raised. |
| **Batch report** | Pick a period and filters: trend and alerts of the batch, as an HTML report (printable to PDF). |
| **Dashboard** | Volume and distribution over time, per category, with a projection and its uncertainty band. |

## Run

```bash
cd ticket_ai
pip install -r requirements.txt
# put your extract (CSV or XLSX) in data/
streamlit run frontend/app.py
```

The extract is read from `data/` (the first `.csv` / `.xlsx` found) and reloaded when the file changes. Nothing is uploaded
in the app. Form widgets (service, group, site...) are filled with the values found in the file.
Tests: `pip install -r requirements-dev.txt && pytest`.

## The five files

```
backend/                     computation, no Streamlit
  data.py       load + filter the extract. Column names and the label values the alert rules look for: edit here
  analytics.py  KPIs, volumes, projection, growth, alerts, nearest tickets
  ai.py         YOUR code: embed() and extract_fields()
frontend/                    rendering, no computation
  app.py        the three pages and the navigation
  render.py     Plotly figures and the HTML report
```

## What you have to do

1. **`backend/ai.py`**: implement the two functions. Until then, the ticket page says which one is missing and the other
   two pages work normally.
   - `embed(texts) -> array (n, dimension)`: called once on the whole history (cached), and on each new ticket.
   - `extract_fields(ticket) -> dict`: returns the LLM columns for a new ticket, e.g.
     `{"IncidentType": ..., "Tone": ..., "Scope": ..., "AffectedSystem": ..., "RootCause": ...}`.
2. **`backend/data.py`**: check the constants at the top against your data. The alert rules compare your columns with
   `SLA_BREACHED`, `NEGATIVE_TONES`, `WIDE_SCOPES` and `SECURITY`; these are placeholders until you set your own labels.
   `Temps de resolution` is assumed to be in hours.

Headers are matched ignoring accents, case and spacing. Required columns: `Ouvert le`, `Description courte`, `Description`;
any other missing column is added empty.

## How it works

**Similar tickets**: `similarity = 0.7 x cosine(embeddings) + 0.3 x share of matching categories` (service, group,
incident type, affected system, site; the last two come from your LLM). Both terms are in [0, 1], so 0.8 reads as close
wording and mostly the same context. A brute-force dot product is milliseconds for hundreds of thousands of tickets, so no
vector database. Ties go to the most recent ticket.

**Ticket alerts** are plain rules in `analytics.ticket_alerts`: urgency and impact both at the top level, security type,
wide scope, negative tone, 5+ similar tickets in the last 14 days (error at 10+), and similar tickets that were hard to
resolve (SLA missed or reopened).

**Batch alerts** (`analytics.batch_alerts`): volume spike against the previous weeks, sustained rise over 4 weeks, and
topics (type, system, root cause) that grew at least 1.5x, hold 15%+ of the total increase and are unlikely to be noise
(Poisson test).

**Projection**: damped Holt exponential smoothing with an approximate 95% band that widens with the horizon. Seasonality is
ignored on purpose: one or two years of weekly data are too few to estimate it. Partial first and last periods are dropped.

## Notes

- **The HTML report embeds Plotly's JavaScript (about 4.7 MB)** so it opens offline. For a PDF, use its
  "Print / save as PDF" button (or Ctrl+P): charts print as they are.
- **Ticket data is sensitive.** Whether `data/` is committed is your call; add `data/*.csv` and `data/*.xlsx` to `.gitignore`
  if it should not be.
- Theme: `.streamlit/config.toml` (Thales palette). Streamlit reads it from the folder you launch from.
- Tested with Streamlit 1.65, pandas 3, Python 3.12.
