# 5. Streamlit application

```bash
uv run itsm app          # http://localhost:8501
```

The app is the visual face of the project: the same library, the same models,
the same numbers as the CLI and the API - rendered, in interactive Plotly, for
someone who will not read a JSON report.

---

## 5.1 Architecture

```
itsm/app/
├── main.py        # entrypoint: page config, sidebar, st.navigation
├── state.py       # every cached resource (data, models, scans)
├── charts.py      # Plotly figure builders, no Streamlit calls
└── views/         # one module per page, each exposing render()
    ├── overview.py       trends, backlog, status updates
    ├── families.py       discovered families vs declared taxonomy
    ├── embedding_map.py  2D ticket map (UMAP + HDBSCAN), tuned live
    ├── signals.py        spikes and emerging weak signals
    └── triage.py         the ITSM intake form scored live
```

### Choices

**The app contains no analysis.** Every page calls `itsm.analysis`,
`itsm.models` or `itsm.pipelines`. A metric shown in the app is computed by the
function the API serves and the tests cover; the three surfaces cannot drift.
This is the same rule stated in [ADR 0001](adr/0001-stack-and-baselines.md).

**Caching is explicit and layered.** Streamlit reruns the whole script on every
interaction, so `state.py` is the single place where that cost is controlled:

| Decorator | Used for | Why |
|-----------|----------|-----|
| `st.cache_resource` | models, artifact store, corpus matrix | Unhashable, process-wide singletons; a model must be deserialised once. |
| `st.cache_data` | tickets, events, scans, reference tables | Serialisable results, keyed by arguments - moving a threshold slider re-runs only the detection. |

**Charts are pure functions returning Plotly figures.** `charts.py` builds
`go.Figure` objects and never touches `st.*`, so a chart can be reused,
composed, or asserted on in a unit test (trace count, trace names, encoded
values) rather than only eyeballed.

Plotly rather than a static renderer because of what these charts are *for*.
Nobody looks at the anomaly chart to admire the curve; they look at it to ask
"how far above normal was that Tuesday, and what came in that day?". That
question needs hover, zoom and click, and answering it in a picture means
adding a second chart instead. Concretely, the app gets:

| Interaction | Where it earns its place |
|---|---|
| Hover with a unified tooltip | Reads observed **and** expected **and** the alert threshold for one period at once - the three numbers a judgement needs. |
| Legend click / double-click | Hides or isolates a family on the split volume chart and on the 2D map, which is how you check whether one category drives a trend. |
| Drag-to-zoom, range presets (1m/3m/6m/all), range slider | A year of daily points is unreadable at full extent; the presets move from the year to the week of the incident without a page reload. |
| **Click a point on the anomaly chart** | Lists that period's actual tickets underneath (`on_select="rerun"`). This is the step from "a score says something happened" to "here is what came in". |
| WebGL rendering on the 2D map | A few thousand embedded tickets stay smooth to pan and zoom. |

The cost is one real dependency (`plotly`, itself only `narwhals` + `packaging`)
where Altair came bundled with Streamlit. That is the trade accepted: the charts
are the product here, not decoration.

**Chart width follows the modern Streamlit API.** `width="stretch"` is the
default, so the deprecated `use_container_width=True` appears nowhere; this is
why the project floor is `streamlit>=1.63`.

**Pages are declared, not discovered.** `st.navigation` with explicit
`st.Page(...)` entries instead of a magic `pages/` folder: the app stays a
normal importable package that `ruff` and `mypy` check like the rest.

**Training is a button, never a side effect of a page load.** When artifacts
are missing the app stops and offers *Prepare the demo*, which runs the build +
training pipelines inside `st.status` and then clears the caches. A page render
must never silently overwrite a model.

---

## 5.2 The pages

### Overview - what the service desk looks like

Headline indicators (volume, backlog, MTTR median/p90, SLA breach, first-time
right), the arrival curve at day/week/month granularity, optionally split by
category, the **arrivals vs departures** chart with the running backlog on a
second axis, backlog ageing, a per-dimension indicator table, and the
status-update block: transition matrix, dwell time per state, rework rates, and
a *follow one ticket* expander showing a single ticket's full chronology.

### Ticket families - what people actually report

The clusters discovered from the designations, each named by class-based
TF-IDF, with their size, their dominant declared category and their purity.
Selecting a family shows its discriminative terms and representative tickets.
The heatmap compares discovered families against declared categories - the
mismatches are the actionable finding, not the agreement.

### Ticket map - the corpus as a picture

A companion to *Ticket families*: sentence embeddings (or TF-IDF + SVD offline)
over designation **and** description, projected to 2D with UMAP and clustered
with HDBSCAN **in that same 2D space** - so the scatter the analyst reads is
exactly the space the clusters were found in, not a flattened shadow of a
higher-dimensional model. Sample size, `n_neighbors`, `min_dist` and
`min_cluster_size` are tuned live; nothing is persisted, because the point is
exploration rather than serving.

Colour switches between the discovered cluster and any declared attribute
(category, channel, priority, assignment group, language), which turns the
mismatch between what users report and how tickets are filed into something you
can see rather than infer from a table. HDBSCAN's noise label is kept and
greyed out: tickets that belong to no dense family are a finding, not an error.

### Trends & weak signals - what is about to break

Series selector (category, assignment group, priority, site), day/week step,
and **two threshold sliders that re-run the detection live**: the spike
threshold (robust z) and the emerging threshold (EWMA). Below the alert table,
the evidence: observed volume against the expected baseline, the **spike
threshold drawn as a dotted line** (`expected + z x scale`, so the decision rule
is visible rather than implicit), alert markers by kind, and the EWMA of
residuals against its decision line.

**Clicking any point on that chart lists the tickets opened in that period for
that series.** An anomaly score says *something* happened; the ticket list says
*what*, and it is the difference between a chart that impresses and one that
gets used.

On the curated dataset the page goes one step further and **scores the
detector** against the injected anomalies: recall, precision, false alarms per
100 series-days and, above all, mean detection delay. Moving a slider and
watching the delay trade against the false-alarm rate is the most direct way to
show that threshold choice is a business decision.

### Intake & triage - the form

An ITSM intake form (designation, description, channel, site, declared
category, impact/urgency with the ITIL priority derived from them), with four
prefilled bilingual examples. On submit, every model runs on it:

* **routing** - predicted category, confidence, suggested assignment group, and
  the class probability bars. When confidence falls below the threshold the
  page says *route to human triage* instead of pretending; when the declared
  category differs from the prediction, it flags the likely misroute;
* **sentiment** - label, signed score, the matched markers as evidence, and an
  escalation hint when the requester is frustrated;
* **family** - the discovered cluster, its terms and its share of volume;
* **comparable tickets** - the five nearest past designations in the same
  vector space, with how they were routed and how long they took, plus the
  median/p90 resolution time of the predicted category as an ETA.

That last block is the bridge from "a model outputs a label" to "an agent gets
a decision they can act on".

---

## 5.3 Limits

* The app reads the extract loaded at start-up; it is a decision-support view
  over a curated dataset, not a live console. Pointing it at a warehouse means
  changing `state.tickets()` only.
* It has no authentication. Ticket text is personal data: put it behind the
  same gateway as the API before exposing it beyond a demo.
* Submitting the form scores a ticket, it does not create one. Writing back to
  an ITSM tool is deliberately out of scope - that integration belongs to the
  tool's own API and its approval workflow.
