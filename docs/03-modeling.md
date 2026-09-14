# 3. Modelling

Three model families, three evaluation protocols. The rule applied throughout:
**a model is stored together with the metrics it obtained**, and those metrics
are exposed by the API (`GET /v1/models`). A score that cannot be inspected at
serving time is a score nobody will check.

---

## 3.1 Category classification

`itsm/models/classifier.py`, `itsm/pipelines/train_classifier.py`.

**Task.** Predict the routing category from the designation. Value: cut the
misrouting that generates reassignments.

### Choices

*Linear over n-grams as the default.* On short, templated, domain-specific
texts with tens of thousands of examples, a calibrated `LinearSVC` over
word+char TF-IDF lands within a couple of macro-F1 points of a fine-tuned
transformer, trains in seconds on CPU, and stays inspectable
(`top_features_for(label)` returns the n-grams driving a class). The
transformer path exists behind `backend="embeddings"`, and the pipeline scores
both on the same split, so the trade-off is measured rather than assumed.

*Probabilities, not labels.* `LinearSVC` has no `predict_proba`, so it is
wrapped in `CalibratedClassifierCV`. Routing is only automated above a
confidence threshold; below it, `predict_with_confidence` returns
`abstained=True` and the ticket goes to human triage. This is the difference
between a demo and something a service desk will switch on.

*Imbalance is structural.* Password resets outnumber database incidents by two
orders of magnitude. Classes are weighted and the headline metric is
**macro-F1**, never accuracy.

### Evaluation protocol

Two splits, on purpose:

* **`temporal`** (default) - train on the past, test on the most recent 20%.
  Lower and more honest: ticket vocabulary drifts (new applications, new error
  codes, a migration that renames everything). Classes absent from the training
  window are dropped from the test set rather than counted as free errors.
* **`stratified`** - random, class-proportional. Measures raw separability;
  use it to compare representations, not to forecast production accuracy.

The gap between the two split strategies is itself the drift estimate.

Reported:

| Metric | Why |
|--------|-----|
| `macro_f1` | Headline. Small families count as much as large ones. |
| `balanced_accuracy` | Cross-check on macro-F1. |
| `top3_accuracy` | A triage UI can propose three categories; this is the achievable assist rate. |
| `per_class` + `worst_classes` | *Which* families fail - usually the small ones, or two that genuinely overlap. |
| `confusion_matrix` | Confusions between two categories often mean the taxonomy, not the model, is the problem. |
| `coverage_curve` | **The decision-making curve**: accuracy restricted to tickets above each confidence threshold, with the share of volume covered. It answers "what fraction can we auto-route at 95% accuracy?". |
| `model_comparison` | `majority`, `complement_nb`, `logreg`, `linear_svc` on the same split. A macro-F1 that does not clearly beat `majority` means the pipeline is broken, not that the task is hard. |
| `cross_validation` (`--cv`) | 5-fold macro-F1 mean and std, to size the uncertainty of a single split. |

Measured on the reference dataset: macro-F1 0.864 against a 0.027 majority
baseline, and ~90% of the volume auto-routable at ~93% accuracy above a 0.8
confidence threshold. Three very different estimators land within one point of
each other, which says the ceiling is the label quality, not the model - see
[`06-results.md`](06-results.md).

### Extending

The same pipeline trains on `--label assignment_group` (routing target) or
`--label priority` (priority prediction, useful to detect under-prioritised
tickets) without any code change.

---

## 3.2 Sentiment / frustration

`itsm/models/sentiment.py`, `itsm/pipelines/eval_sentiment.py`.

**Task.** Detect requester frustration to enable proactive handling.

### The reframing that matters

Product-review sentiment does not transfer: nearly every ticket is "negative"
in the everyday sense, because something is broken. The operationally useful
signal is **escalation risk** - blocked, already chased twice, writing in
capitals - not polarity.

### Backends

* **`lexicon` (default)** - a compact FR/EN lexicon with ITSM markers
  ("relance", "troisieme fois", "totalement bloque", "escalate"), negation and
  intensifier handling, plus shouting and exclamation detection. No download,
  fully auditable, and it returns the matched `evidence` with every score -
  which is what makes a human trust or overrule it.
* **`transformer`** - a multilingual model from the hub
  (default `cardiffnlp/twitter-xlm-roberta-base-sentiment`), behind the `[hf]`
  extra.

### The honest caveat

**There is no gold sentiment annotation on ITSM tickets.** Scoring uses a weak
label:

* on synthetic data, the mood that generated the wording;
* on real data, an operational proxy (`operational_weak_labels`): reopened, or
  SLA-breached with at least two reassignments - which is what a stakeholder
  means by "an unhappy requester" anyway.

A weak label is noisy and correlates with variables the model never sees. The
resulting numbers are valid for **comparing backends on equal footing** and for
catching gross failures; they are *not* an absolute accuracy, and this report
says so in its own `label_caveat` field.

The path to a defensible figure is a few hundred human annotations.
`sample_for_annotation()` builds that batch, prioritising the tickets where the
backends disagree - the cheapest proxy for difficulty - and filling the rest at
random to keep the sample representative. Reported metrics include
**Cohen's kappa** rather than raw accuracy, because a "always neutral" model
scores well on accuracy alone.

Measured: macro-F1 0.804, kappa 0.664, and an informative asymmetry - 87%
recall on frustrated requesters at 61% precision. For flagging tickets for a
proactive callback that is the right side to err on, but it is a finding to
state, not a number to round up.

---

## 3.3 Weak-signal detection

`itsm/models/weak_signal.py`, `itsm/pipelines/eval_weak_signal.py`.

**Task.** Flag a family whose arrival rate departs from its own trend *before*
it becomes an obvious outage.

### Why a plain rolling z-score fails

Three properties of ITSM arrival series break it:

1. **Weekly seasonality.** Monday is not an anomaly. The baseline is computed
   over the *same weekday* inside the trailing window (56 days by default).
2. **Counts, not Gaussians.** Daily volumes are small integers; the natural
   noise model is Poisson, whose variance equals its mean. The detector reports
   a Poisson tail probability next to the z-score, so 2 -> 8 tickets is judged
   on the right scale.
3. **Contamination.** A window that already contains the start of an incident
   drags mean and standard deviation towards it, and the detector blinds
   itself. Median and **MAD** are used instead: up to half the window can be
   corrupted before the baseline moves. The scale is floored at
   `sqrt(expected)` so a perfectly flat window cannot produce an infinite
   z-score.

### Two alarm families

| Kind | Rule | Catches |
|------|------|---------|
| `spike` | robust z >= 3.5 **and** Poisson tail p <= 1e-3 | An outage that already happened. |
| `emerging` | EWMA of positive standardised residuals >= 1.1 | **The weak signal**: several consecutive mild deviations, none individually significant. Fires days before the spike. |

The EWMA accumulates only positive deviations: a quiet week must not build up
credit that masks the next rise.

### Evaluation - the part that is usually missing

Anomaly detectors are normally shipped unevaluated ("look, it found the
outage"). Here the generator injects a known set of spikes, ramps and level
shifts, so the detector is scored like a classifier:

| Metric | Why it matters |
|--------|----------------|
| `recall` | Share of injected anomalies caught at all. A detector that misses the outage is worthless regardless of precision. |
| `precision` | Share of alerts that fall inside a true window. |
| `false_alarms_per_100_series_days` | What an on-call team actually experiences. Precision alone is misleading: it depends on how long an anomaly lasts. |
| **`mean_detection_delay_days`** | Days between the start of the anomaly and the first alert. **This is the metric that separates a weak-signal detector from a post-mortem tool.** |
| `per_anomaly` detail | Per injected event: detected or not, when, by which alarm kind. A detector that catches the two spikes and misses the slow ramp has failed at the actual task. |

`threshold_sweep()` grid-searches `z_spike` x `ewma_threshold` and returns the
operating points. There is no universally best setting: it is a business
trade-off between catching an incident a day earlier and waking someone up for
nothing. The sweep makes that explicit instead of hiding it in a constant.

On a real extract with no ground truth, the same pipeline runs unscored: it
produces the alerts and the diagnostic series, and thresholds are set from the
false-alarm budget the team accepts.

### Measured behaviour

Detection rates, delays and the threshold ablation that set the defaults are in
[`06-results.md`](06-results.md). Headline: all four injected anomalies are
caught on the reference dataset, the slow ramp 12 days in - well before it
peaks - while the permanent level shift takes 62 days, because a rolling
baseline absorbs a step change by construction. Across five seeds recall sits
at 0.95-1.00 depending on the threshold, and the threshold is a false-alarm
budget rather than a tuning constant.

### Beyond the current implementation

Worth adding when the data justifies it, in rough order of value:

1. **Text-aware signals** - run the detector on *cluster* series rather than
   category series, so "a new kind of ticket appearing" is detectable even
   inside a stable category volume. The clusterer already supports it
   (`--by cluster` after joining assignments).
2. **Multivariate confirmation** - correlate a volume alert with a jump in
   reopen rate or negative sentiment; a single-metric alert is easy to dismiss.
3. **STL decomposition or a Bayesian structural time-series** once there are
   two full years of history, which is what a yearly seasonal term needs.
