# 4. API

`itsm/api/`, served with `itsm serve` (uvicorn). Interactive documentation at
`/docs`, OpenAPI schema at `/openapi.json`.

## 4.1 Principles

**The API is a backbone, not a second implementation.** Every endpoint calls
the same library function the CLI and the Streamlit app call. There is exactly
one implementation of "compute the MTTR" or "score a text", so the three
surfaces cannot drift apart.

**Nothing is loaded at import time.** The app starts instantly and models are
deserialised lazily on the first request that needs them
(`itsm/api/deps.py`, `lru_cache`). A service with no trained artifact still
starts and still serves the statistics endpoints; `/readyz` reports what is
missing, and model endpoints answer **503 with the command to run**, not a
stack trace.

**Metrics travel with the model.** `GET /v1/models` returns each artifact's
training date, dataset, parameters *and* evaluation scores. A consumer can
always find out what the prediction it just received is worth.

**Every response is traceable.** A middleware attaches `x-request-id` (echoing
the caller's, or generating one) and `x-process-time-ms`, and the unhandled
exception handler returns a uniform envelope carrying that id instead of
leaking internals.

## 4.2 Endpoints

### Service

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/healthz` | Liveness. Always `ok` if the process answers. |
| `GET` | `/readyz` | `degraded` when an artifact is missing; lists which. |
| `GET` | `/v1/models` | Artifacts with metadata and metrics. |

### Inference

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/classify` | Body: `{"texts": [...], "min_confidence": 0.6, "top_k": 3}`. Returns label, confidence, the `top_k` alternatives and **`abstained`** - true means "send to human triage", and clients are expected to branch on it. |
| `POST` | `/v1/sentiment` | Returns label, signed score in `[-1, 1]` and the matched `evidence` markers. |
| `POST` | `/v1/clusters/assign` | Assigns designations to the families discovered by the analysis. |

### Privacy

| Method | Path | Notes |
|--------|------|-------|
| `POST` | `/v1/anonymise` | Detects personal data and replaces each value with its entity tag (`<PERSON>`, `<EMAIL>`, `<IBAN>`). The response returns entity **types and offsets but never the matched values** - an API response is logged, cached and screenshotted, and echoing the personal data back would recreate the exposure the call exists to remove. |

Exposed so a caller can scrub text *before* sending it anywhere else,
including before the other endpoints of this very API. A service that
anonymises its own storage while accepting raw personal data over the wire and
writing it to a request log has moved the problem, not solved it.

Batches are capped at 512 texts per request (Pydantic validation, `422`
otherwise): a bounded batch keeps latency predictable and makes back-pressure
the caller's problem, not the service's.

### Analytics

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/v1/stats/overview` | Headline indicators. |
| `GET` | `/v1/stats/by/{dimension}` | Any categorical column; `404` on an unknown one. |
| `GET` | `/v1/stats/flow?freq=W` | Volume, backlog, ageing, transitions, dwell times, rework. |
| `GET` | `/v1/tickets/{id}/trace` | One ticket and its full status history. |
| `GET` | `/v1/clusters` | Discovered families with terms and representatives. |
| `POST` | `/v1/signals/scan` | Spikes and emerging weak signals; thresholds are request parameters, so a caller can tighten them without a redeploy. |
| `GET` | `/v1/signals/series` | Observed vs expected counts and scores, day by day - the evidence behind an alert. An anomaly score without its series is not actionable. |

## 4.3 Example

```bash
curl -s localhost:8000/v1/classify -H 'content-type: application/json' -d '{
  "texts": ["Le VPN se deconnecte toutes les cinq minutes en teletravail"],
  "min_confidence": 0.6
}' | jq
```

```json
{
  "model_name": "linear_svc",
  "predictions": [{
    "text": "Le VPN se deconnecte toutes les cinq minutes en teletravail",
    "label": "Network & VPN",
    "confidence": 0.91,
    "abstained": false,
    "alternatives": [
      {"label": "Network & VPN", "score": 0.91},
      {"label": "Access & Accounts", "score": 0.04},
      {"label": "Hardware & Workstation", "score": 0.02}
    ]
  }]
}
```

## 4.4 Data access

The analytics endpoints read the ticket extract the service was started with,
cached in memory for the process lifetime. That is the right shape for a
demonstrator and for a nightly-refreshed read model; it is **not** the right
shape for a live ITSM database.

For production, replace the two dataframe dependencies in `itsm/api/deps.py`
(`get_tickets`, `get_ticket_events`) with a repository querying the warehouse
or a materialised view. No router changes: they only ever receive a dataframe.

## 4.5 Deliberately not included

These belong to the deployment platform, not to a model service, and hard-coding
one choice here would be wrong in most target environments:

* **Authentication / authorisation** - add an API-key or OIDC dependency on the
  routers (`dependencies=[Depends(verify_token)]` at `include_router` level).
  Ticket content is personal data: in a real deployment this is the first thing
  to add.
* **Rate limiting** - belongs at the gateway.
* **Metrics export** - a Prometheus ASGI middleware; the per-request timing is
  already computed and logged.
* **Model reloading** - artifacts are cached for the process lifetime;
  `reset_caches()` exists for tests, and the production pattern is to redeploy
  rather than hot-swap a model under live traffic.
* **Async workers** - the models are CPU-bound and fast (milliseconds for the
  linear classifier). Under a transformer backend, move inference to a worker
  pool or a dedicated inference server rather than blocking the event loop.
