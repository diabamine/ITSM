# 6. Measured results

Everything below was produced on the default configuration -
`ITSM_DATASET=synthetic`, 8 000 target tickets over 365 days, `seed=42` - and
is reproducible with:

```bash
uv run itsm data build --force
uv run itsm privacy anonymise && uv run itsm privacy eval --backend rules --backend transformer
uv run itsm analyze designations && uv run itsm analyze lifecycle
uv run itsm train classifier && uv run itsm eval sentiment && uv run itsm eval signals
uv run python scripts/summarize_run.py
```

> **Read these as protocol validation, not as production forecasts.** They are
> measured on generated data whose difficulty the generator itself sets (7%
> label noise, 13% ambiguous wordings, 12% typos, and planted personal data).
> What transfers to a real extract is the *method* and the *shape* of the
> trade-offs, not the absolute numbers. See
> [ADR 0001](adr/0001-stack-and-baselines.md).

---

## 6.1 Dataset

| | |
|---|---|
| Tickets / status updates | 7 897 / ~48 000 |
| Period | 365 days, 21.7 tickets/day |
| Categories / languages | 12 / FR 60%, EN 40% |
| Open at extraction | 36 (0.5% of annual volume) |
| MTTR median / p90 | 15.1 h / 65.5 h |
| SLA breach rate | 33.8% |
| Reopen rate | 5.8% |
| First-time-right | 57.6% |
| **Tickets carrying personal data** | **52.9%** (9 448 planted identifiers) |

The open backlog is small because it is a *stock* against a year of *flow*: at
~22 tickets/day and a median resolution under a day, a few dozen tickets in
flight is the correct steady state. The long tail (3% of tickets stuck 6-30x
their normal handling time) is what populates the ageing view.

## 6.2 Classification

Temporal split, 6 317 train / 1 580 test, TF-IDF word+char, calibrated
`LinearSVC`.

| Model | macro-F1 | accuracy | balanced acc. | top-3 |
|-------|----------|----------|---------------|-------|
| `complement_nb` | **0.868** | 0.885 | - | 0.951 |
| `linear_svc` (default) | 0.864 | 0.883 | 0.872 | 0.951 |
| `logreg` | 0.863 | 0.882 | - | 0.951 |
| `majority` (baseline) | 0.027 | 0.187 | - | 0.310 |

* **The ceiling is the data, not the model.** Three very different estimators
  land within half a point of each other, and ~0.86 is roughly what 7% label
  noise plus 13% ambiguous wordings allow. When every estimator agrees, more
  modelling is the wrong investment - better labels are the right one.
* **Weakest classes**: Database & Middleware (F1 0.72), Mobile Devices (0.79),
  Storage & File Shares (0.83) - the smallest families, and the ones sharing
  the "application slow" / "cannot open the file" wordings.
* Naive Bayes edges ahead, inside the noise of a single split; `linear_svc`
  stays the default because it calibrates better, which is what the abstention
  rule below depends on.

### The coverage curve - the number a service desk decides on

| Confidence threshold | Volume covered | Accuracy on covered |
|---|---|---|
| 0.0 (no abstention) | 100% | 88.3% |
| 0.4 | 99.1% | 88.8% |
| 0.6 | 92.6% | 92.6% |
| 0.8 | 90.1% | **93.2%** |
| 0.9 | 31.9% | 94.4% |

Auto-routing above 0.8 confidence handles ~90% of the volume at ~93% accuracy
and sends the rest to human triage. Pushing to 0.9 buys 1.2 points of accuracy
and loses 58 points of coverage - a bad trade, and exactly the decision this
curve exists to make visible.

## 6.3 Ticket families

Auto-selected k = 36 by silhouette (candidate range 12-36).

| Metric | Value |
|---|---|
| Silhouette / Davies-Bouldin | 0.45 / 1.64 |
| Homogeneity / completeness | 0.52 / 0.35 |
| Adjusted Rand / NMI | 0.15 / 0.42 |
| Mean cluster purity | 0.62 |

**Homogeneity high, completeness low** is the signature of clusters *finer*
than the taxonomy, and it is the expected outcome: 12 declared categories
against 12 × ~4.5 symptom templates × 2 languages of natural granularity. A low
ARI here is a measurement, not a failure - it quantifies that the routing
taxonomy is coarser than what users report.

Silhouette rises monotonically over the candidate range, so the automatic
selection lands on its upper bound: **the range is the real prior**, and an
analyst should set `--k` from the purity/completeness trade-off rather than
trust the selection blindly. Stated rather than hidden.

## 6.4 Sentiment

Lexicon backend against the generator weak label, 4 000 tickets.

| Metric | Value |
|---|---|
| macro-F1 | 0.804 |
| Accuracy | 0.845 |
| Cohen's kappa | 0.664 |
| Negative recall | 0.871 |
| Negative precision | 0.606 |

The asymmetry is the finding: the lexicon **catches 87% of frustrated
requesters while 39% of its alerts are false**. For flagging tickets for a
proactive callback that is the right side to err on, and it follows directly
from scoring politeness formulas near zero while weighting escalation markers
heavily. Raising precision needs real annotations, not more rules.

## 6.5 Weak signals

Default configuration on the reference dataset (12 series × 365 days):

| Injected anomaly | Kind | Detected | Delay | By |
|---|---|---|---|---|
| Network & VPN | spike (×5, 3 d) | yes | **1 d** | `emerging` |
| Email & Collaboration | ramp (×1 → ×3.2 over 25 d) | yes | **12 d** | `emerging` |
| Business Applications | level shift (×1.8) | yes | 62 d | `spike` |
| Security | spike (×7, 2 d) | yes | **1 d** | `spike` |

All four caught; 93 alerts, precision 0.19, 1.7 false alarms per 100
series-days. The level shift is the slow one: a permanent ×1.8 step is, by
construction, absorbed into the trailing baseline, and it only trips the
detector later on a random fluctuation. That is the documented weakness of a
rolling-baseline method and the reason a changepoint layer is on the list in
[`03-modeling.md`](03-modeling.md).

Note the series are grouped by the **recorded** category, which carries 7%
label noise - so a share of each anomaly's tickets is scattered into other
families before the detector ever sees them. That is deliberately the realistic
setting: you detect on the data you have.

### Threshold ablation (5 seeds, 9 000 tickets, 330 days)

| Guard band | EWMA threshold | Recall | Precision | Mean delay | False alarms /100 series-days |
|---|---|---|---|---|---|
| 0 d | 0.7 | 1.00 | 0.14 | 2.3 d | 6.0 |
| 0 d | 0.9 | 1.00 | 0.19 | 2.6 d | 3.5 |
| 0 d | 1.1 | 1.00 | 0.25 | 6.0 d | 2.0 |
| 0 d | 1.4 | 0.95 | 0.40 | 10.0 d | 0.8 |
| 7 d | 0.7 | 1.00 | 0.15 | 2.1 d | 6.0 |
| 7 d | 0.9 | 1.00 | 0.21 | 2.6 d | 3.4 |
| **7 d** | **1.1** (default) | 0.95 | 0.27 | 6.4 d | 2.0 |
| 7 d | 1.4 | 0.95 | 0.39 | 10.6 d | 0.9 |

Conclusions actually supported by this table:

* **The EWMA threshold is the business dial.** 0.9 detects ~4 days earlier and
  nearly doubles the false alarms; 1.4 halves them and costs another 4 days.
  There is no correct value, only an accepted false-alarm budget - which is why
  `threshold_sweep()` ships with the model and the app exposes the slider.
* **The guard band is *not* proven.** An earlier measurement on a different
  random stream showed it improving precision at no recall cost; this one shows
  it costing one anomaly out of twenty at the default threshold. Five seeds
  cannot separate those. It stays enabled because the mechanism it prevents (a
  slow ramp walking into its own baseline) is real and its downside here is one
  marginal detection - but it should not be quoted as a measured win, and that
  correction is recorded rather than quietly dropped.

## 6.6 Anonymisation

Scored on 2 000 tickets, 2 087 annotated personal values. Full discussion:
[`07-privacy.md`](07-privacy.md).

| | rules | transformer |
|---|---|---|
| Coverage recall | **1.000** | 0.999 |
| Exact-boundary recall | **1.000** | 0.996 |
| Precision | **0.999** | 0.976 |
| Leak rate, direct identifiers | **0.000** | 0.0012 |
| **Residual PII in the output** | **0 / 2 087** | 2 / 2 087 |
| False-positive spans | **2** | 50 |
| Utility cost (macro-F1) | **0.000** | −0.001 |

And the number that actually decides deployment - recall on twelve names
neither backend has ever seen:

| Backend | Recall on unseen names |
|---|---|
| rules | **0.667** |
| transformer | **0.944** |

The headline table flatters the rule engine because the corpus draws its names
from the very list its gazetteer holds. On names outside any list, the rule
engine misses a third of them; the transformer misses one in eighteen.

## 6.7 Quality gates

```
90 tests passed, 3 slow (pytest)
ruff check       All checks passed
ruff format      61 files formatted
mypy             no issues found in 60 source files
```

The suite includes the Streamlit pages rendered end to end through
`streamlit.testing`, the API exercised through `TestClient`, and - for the
anonymisation - an end-to-end assertion that no direct identifier survives in
the rewritten corpus.
