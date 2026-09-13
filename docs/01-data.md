# 1. Data

## 1.1 The problem with public ITSM data

ITSM analytics needs three things at once: **free text** (what the user typed),
**a taxonomy** (what it was classified as) and **an audit trail** (how the
ticket moved). No public corpus provides all three, because the first two are
confidential and the third is only meaningful with the first two.

| Source | Free text | Taxonomy | Audit trail | Verdict |
|--------|-----------|----------|-------------|---------|
| [UCI 498 - incident management event log](https://archive.ics.uci.edu/dataset/498/incident+management+process+enriched+event+log) | anonymised codes (`Symptom 123`) | anonymised codes (`Category 26`) | **yes** - 141 712 events, 24 918 incidents, reassignments, reopens, SLA flag | lifecycle and time-series work; no NLP |
| [`Tobi-Bueck/customer-support-tickets`](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) | **yes**, multilingual | **yes** (`queue`, `priority`, `type`) | no | NLP works; no flow, no time series |
| Kaggle IT-support exports | yes | yes | no | same limitation |

### Decision

Support all of them through one schema, and add a **seeded synthetic
generator** as the default so that the whole project is runnable offline, on
day one, with ground truth for the parts that would otherwise be
unevaluable.

This is not a shortcut around real data - it is what makes the evaluation
pipelines meaningful. The generator knows which tickets belong to which
family (ground truth for clustering), which mood produced which wording
(ground truth for sentiment), and **when it injected an anomaly** (ground truth
for weak-signal detection, which is otherwise impossible to score).

The moment a real extract is available, point `ITSM_DATASET=csv` at it: the
pipelines do not change, only the evaluation of the synthetic-only parts is
replaced by the protocols described in [`03-modeling.md`](03-modeling.md).

## 1.2 Canonical schema

Two tables, both defined in `itsm/data/schema.py` with the ServiceNow field
names used by the UCI dataset.

**Tickets** - one row per ticket, final state and aggregates:

| Field | Notes |
|-------|-------|
| `ticket_id`, `opened_at`, `resolved_at`, `closed_at` | `resolved_at`/`closed_at` are null for open tickets. |
| `designation` | The short description typed at intake - the main NLP input. |
| `description`, `language` | Body and language code (`fr`, `en`). |
| `category`, `subcategory`, `assignment_group` | The routing taxonomy - the classification target. |
| `priority`, `impact`, `urgency` | ITIL priority derived from impact x urgency. |
| `channel`, `site` | Portal / email / phone / chat, and location. |
| `state` | `New`, `Assigned`, `In Progress`, `Awaiting User Info`, `Resolved`, `Closed`. |
| `reassignment_count`, `reopen_count`, `sys_mod_count` | Rework counters from the audit system. |
| `sla_breached`, `resolution_hours` | Outcome measures. |
| `frustration_label` | **Weak label**, synthetic only, used exclusively to score sentiment models. Never a model input. |

**Events** - one row per status update: `ticket_id`, `event_index`,
`timestamp`, `state`, `assignment_group`, `is_reassignment`, `is_reopen`.
This table is what makes dwell times, transition matrices and rework analysis
possible; without it "status updates" can only be counted, not analysed.

`validate_tickets()` / `validate_events()` coerce any source into this shape,
fill schema defaults for missing optional columns and fail loudly on missing
required ones.

## 1.3 What the generator simulates

`itsm/data/synthetic.py`, driven by the catalog in `itsm/data/catalog.py`
(12 families, FR/EN templates with slot filling, assignment groups, per-family
priority profiles and handling times).

**Arrivals** - a Poisson process whose daily intensity is the product of:

* a family weight (access & accounts is 17% of volume, databases 4%);
* a weekday profile (Monday +30%, Saturday -86%);
* a yearly profile (August trough, September peak);
* a growth trend (+12%/year by default);
* per-family effects (password resets are x1.9 on Mondays);
* **injected anomalies** (see below).

Arrival hours follow a business-hour profile with a lunch dip.

**Lifecycle** - a state machine per ticket: assignment delay and handling time
are log-normal, scaled by priority; 8% of tickets are misrouted and generate an
extra reassignment; 25% go through an `Awaiting User Info` phase; 3% get stuck
6-30x longer than normal (the vendor/spare-part/silent-user tail every desk
carries); reopens are drawn with a probability that grows with the number of
reassignments. Anything that would resolve after the extraction horizon stays
**open**, which reproduces the backlog a real extract contains - a few dozen
tickets in flight against a year of volume, since backlog is a stock and volume
is a flow.

**Wording and mood** - the requester mood is drawn from how badly the ticket
went (reopened, breached, reassigned, critical), and *then* determines the
phrasing. That direction matters: the model must recover the mood from the
text alone, and the label is not a function of any feature it sees.

**Difficulty** - this is the part that makes the evaluation mean anything.
Drawn from clean, disjoint templates, the dataset is perfectly separable and
*every* estimator reaches a macro-F1 of 1.0, which measures nothing. Three
mechanisms put a realistic ceiling on it:

| Mechanism | Default | What it reproduces |
|-----------|---------|--------------------|
| `ambiguity_rate` | 13% | Wordings that genuinely belong to several families ("Impossible de me connecter ce matin", "Application tres lente"). Sampled from `AMBIGUOUS_TEMPLATES`, where each entry declares its candidate categories. |
| `label_noise` | 7% | The agent files the ticket under the wrong category. The recorded category *and* assignment group follow the mistake and the ticket gets an extra reassignment - which is exactly what a misroute looks like in a real extract. |
| `typo_rate` | 12% | Transposed or dropped characters, shouted titles, truncations. What the character n-gram channel exists to absorb. |

Set all three to 0 for the noiseless variant. With the defaults, the
classifier tops out near 0.85 macro-F1 - see
[`06-results.md`](06-results.md).

**Injected anomalies** - the manifest returned alongside the data:

| Family | Kind | Shape | What it tests |
|--------|------|-------|---------------|
| Network & VPN | `spike` | x5 over 3 days | Detection of an abrupt outage. |
| Email & Collaboration | `ramp` | linear x1 -> x3.2 over 25 days | **The weak signal**: is it caught early, or only at the peak? |
| Business Applications | `level_shift` | permanent x1.8 | A regression that becomes the new normal - the case a rolling baseline silently absorbs. |
| Security | `spike` | x7 over 2 days | A very short, very sharp event. |

## 1.4 Reproducibility

`generate(seed=...)` fixes the entire dataset. `itsm data build` materialises
it to Parquet under `data/<dataset>/` together with `anomalies.json`; every
later command reads the cache, so an analysis and the model trained on it refer
to exactly the same rows. `itsm data build --force` rebuilds.
