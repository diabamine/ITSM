# 6. Measured results

Everything below was produced by running the pipelines on the default
configuration - `ITSM_DATASET=synthetic`, 8 000 target tickets over 365 days,
`seed=42` - and is reproducible with:

```bash
uv run itsm data build --force
uv run itsm analyze designations && uv run itsm analyze lifecycle
uv run itsm train classifier && uv run itsm eval sentiment && uv run itsm eval signals
uv run python scripts/summarize_run.py
```

> **Read these as protocol validation, not as production forecasts.** They are
> measured on generated data whose difficulty the generator itself sets (7%
> label noise, 13% ambiguous wordings, 12% typos). What transfers to a real
> extract is the *method* and the *shape* of the trade-offs, not the absolute
> numbers. See [ADR 0001](adr/0001-stack-and-baselines.md).

---

## 6.1 Dataset

| | |
|---|---|
| Tickets / status updates | 7 955 / 47 983 |
| Period | 365 days, 21.9 tickets/day |
| Categories / languages | 12 / FR 60%, EN 40% |
| Open at extraction | 32 (0.4% of annual volume) |
| MTTR median / p90 | 15.3 h / 61.6 h |
| SLA breach rate | 32.9% |
| Reopen rate | 5.6% |
| Reassignment rate | 39.1% |
| First-time-right | 58.5% |

The open backlog is small because it is a *stock* against a year of *flow*:
at ~22 tickets/day and a median resolution under a day, a few dozen tickets in
flight is the correct steady state. The long tail (3% of tickets stuck 6-30x
their normal handling time) is what populates the ageing view.

## 6.2 Classification

Temporal split, 6 364 train / 1 591 test, TF-IDF word+char, calibrated
`LinearSVC`.

| Model | macro-F1 | accuracy | balanced acc. | top-3 |
|-------|----------|----------|---------------|-------|
| `complement_nb` | **0.858** | 0.860 | - | 0.928 |
| `linear_svc` (default) | 0.850 | 0.864 | 0.850 | 0.927 |
| `logreg` | 0.848 | 0.860 | - | 0.928 |
| `majority` (baseline) | 0.026 | 0.185 | - | 0.306 |

Three things to read here:

* **The ceiling is the data, not the model.** Three very different estimators
  land within 1 point of each other, and 0.85 is roughly what 7% label noise
  plus 13% ambiguous wordings allow. When every estimator agrees, more
  modelling is the wrong investment - better labels are the right one.
* **Weakest classes**: Database & Middleware (F1 0.74), Telephony & Meeting
  Rooms (0.79), Mobile Devices (0.80) - the three smallest families, and the
  ones sharing the "application slow" / "no sound" wordings.
* **Naive Bayes marginally wins.** It is kept as a baseline rather than
  promoted: the difference is inside the noise of a single split, and
  `linear_svc` calibrates better, which matters for the abstention rule below.

### The coverage curve - the number a service desk actually decides on

| Confidence threshold | Volume covered | Accuracy on covered |
|---|---|---|
| 0.0 (no abstention) | 100% | 86.4% |
| 0.4 | 97.7% | 87.7% |
| 0.6 | 93.0% | 90.3% |
| 0.8 | 88.9% | **91.4%** |
| 0.9 | 31.3% | 92.0% |

Auto-routing above 0.8 confidence handles ~89% of the volume at ~91% accuracy
and sends the remaining 11% to human triage. Pushing to 0.9 buys 0.6 points of
accuracy and loses 58 points of coverage - a bad trade, and exactly the kind of
decision this curve exists to make visible.

## 6.3 Ticket families

Auto-selected k = 36 by silhouette (candidate range 12-36).

| Metric | Value |
|---|---|
| Silhouette / Davies-Bouldin | 0.43 / 1.39 |
| Homogeneity / completeness | 0.53 / 0.36 |
| Adjusted Rand / NMI | 0.16 / 0.43 |
| Mean cluster purity | 0.66 |

**Homogeneity high, completeness low** is the signature of clusters *finer*
than the taxonomy, and it is the expected outcome: 12 declared categories
against 12 x ~4.5 symptom templates x 2 languages of natural granularity. A low
ARI here is a measurement, not a failure - it quantifies that the routing
taxonomy is coarser than what users report.

Measured granularity trade-off (1 800-ticket sample):

| k | homogeneity | completeness | mean purity | silhouette |
|---|---|---|---|---|
| 12 | 0.24 | 0.25 | 0.43 | 0.15 |
| 24 | 0.41 | 0.33 | 0.61 | 0.31 |
| 30 | 0.50 | 0.36 | 0.61 | 0.37 |

Note that silhouette increases monotonically over the candidate range, so the
automatic selection lands on its upper bound: **the range is the real prior**,
and an analyst should set `--k` from the purity/completeness trade-off above
rather than trust the selection blindly. This is stated rather than hidden.

## 6.4 Sentiment

Lexicon backend against the generator weak label, 4 000 tickets.

| Metric | Value |
|---|---|
| macro-F1 | 0.757 |
| Accuracy | 0.798 |
| Cohen's kappa | 0.579 |
| Negative recall | 0.890 |
| Negative precision | 0.509 |

The asymmetry is the finding: the lexicon **catches 89% of frustrated
requesters but half its alerts are false**. For its intended use - flagging
tickets for a proactive callback - that is the right side to err on, and it is
a deliberate consequence of scoring politeness formulas ("merci d'avance") near
zero while weighting escalation markers ("relance", "troisieme fois") heavily.
Raising precision requires real annotations, not more rules; see
[`03-modeling.md`](03-modeling.md).

## 6.5 Weak signals

Default configuration on the reference dataset (12 series x 365 days):

| Injected anomaly | Kind | Detected | Delay | By |
|---|---|---|---|---|
| Network & VPN | spike (x5, 3 d) | yes | **0 d** | `spike` |
| Email & Collaboration | ramp (x1 -> x3.2 over 25 d) | yes | **14 d** | `emerging` |
| Business Applications | level shift (x1.8) | yes | 25 d | `emerging` |
| Security | spike (x7, 2 d) | **no** | - | - |

84 alerts, precision 0.29, 1.37 false alarms per 100 series-days.

**The Security miss is honest and instructive.** That family averages ~1.3
tickets/day, and in August (seasonal trough) a 7x spike produces 5 tickets
against an expected 2. Poisson says p ≈ 0.05 - genuinely weak evidence, and a
detector that fired here would fire constantly elsewhere. The fix for
low-volume series is not a lower threshold but a coarser aggregation
(`--freq W`) or a wider grouping.

### Threshold ablation (5 seeds, 9 000 tickets, 330 days)

Single-dataset results are noisy, so the defaults were chosen on an aggregate:

| Guard band | EWMA threshold | Recall | Precision | Mean delay | False alarms /100 series-days |
|---|---|---|---|---|---|
| 0 d | 0.7 | 1.00 | 0.14 | 2.6 d | 6.1 |
| 0 d | 0.9 | 1.00 | 0.19 | 4.0 d | 3.3 |
| 0 d | 1.1 | 1.00 | 0.29 | 4.3 d | 1.6 |
| 0 d | 1.4 | 0.95 | 0.40 | 8.5 d | 0.7 |
| **7 d** | 0.7 | 1.00 | 0.14 | 3.0 d | 5.9 |
| **7 d** | 0.9 | 1.00 | 0.20 | 3.9 d | 3.3 |
| **7 d** | **1.1** (default) | **1.00** | **0.30** | **4.9 d** | **1.6** |
| **7 d** | 1.4 | 1.00 | 0.44 | 7.7 d | 0.7 |

Conclusions actually drawn from this table:

* **Recall is 1.00 at the default threshold across seeds** - the seed-42 miss
  above is a single unlucky low-volume draw, not the typical case. Both figures
  are reported rather than the flattering one.
* **The guard band earns its place**: equal or better precision at every
  threshold, and it keeps recall at 1.00 where the no-guard variant drops to
  0.95. Its cost is under a day of detection delay.
* **The EWMA threshold is the business dial**: 0.9 detects a day earlier and
  doubles the false alarms; 1.4 halves them and costs three days. There is no
  correct value, only an accepted false-alarm budget - which is why
  `threshold_sweep()` ships with the model and the Streamlit page exposes the
  slider.

## 6.6 Quality gates

```
47 tests passed  (pytest)
ruff check       All checks passed
ruff format      46 files formatted
mypy             no issues found in 46 source files
```

The suite includes the four Streamlit pages rendered end to end through
`streamlit.testing`, and the API exercised through `TestClient`.
