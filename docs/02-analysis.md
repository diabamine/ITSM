# 2. Analysis

Two independent passes, both run before any model is trained:
`itsm analyze designations` and `itsm analyze lifecycle`.

---

## 2.1 Families of tickets, from the designation

### Why not just use the declared category?

Because the declared category is the *routing* taxonomy, not the *problem*
taxonomy. It is chosen by whoever fills the form, it is stable for years while
the estate changes, and it is often wrong (misrouting is the first cause of
reassignment). Clustering the free text answers a different question: *what do
people actually report?* The gap between the two - measured here as cluster
purity and adjusted Rand index - is itself a finding:

* a cluster split across two categories = a taxonomy that separates what users
  experience as one problem;
* a category split across several clusters = a bucket too coarse to route on;
* a cluster with no dominant category = a family nobody has named yet.

### Pipeline

```
normalise -> vectorise -> reduce -> cluster -> label
```

**Normalise** (`itsm/text/normalize.py`). Ticket titles are short, bilingual and
saturated with identifiers. Hostnames, ticket references, error codes, URLs and
numbers are replaced by placeholder tokens (`<asset>`, `<code>`, `<url>`,
`<num>`): this keeps the *presence* of an error code - informative - while
dropping its unique value, which would otherwise create one vocabulary entry
per ticket. Accents are stripped (users type both `problème` and `probleme`),
and a curated FR/EN stop-word list removes the service-desk filler
("bonjour", "merci", "probleme", "urgent") that appears in every family and
therefore separates nothing. No spaCy, no NLTK download: the pass is
deterministic and offline.

**Vectorise** (`itsm/text/vectorize.py`). Default is the union of word 1-2
grams and **character 3-5 grams**. The character channel is what makes the
representation survive real ticket titles: typos, `Outlook`/`outlok`,
agglutinated product names, and FR/EN code-switching inside one sentence.
The alternative backend is a multilingual sentence-transformer, which groups
paraphrases sharing no vocabulary (`VPN drops` / `connexion coupée`) at ~100x
the CPU cost.

**Reduce**. Truncated SVD to 120 dimensions, then L2 normalisation. KMeans
minimises a Euclidean objective that degrades in sparse high-dimensional space;
after SVD the centroids are meaningful, and on L2-normalised vectors Euclidean
distance is monotone in cosine distance (spherical KMeans).

**Cluster**. KMeans by default: deterministic, cheap to refit, and it yields a
centroid per family that the API reuses to assign a new ticket in
microseconds. `k` is selected by silhouette over a candidate range (on a
subsample - silhouette is quadratic). HDBSCAN is available for exploration: it
finds variable-density families and refuses to assign outliers (label `-1`),
which is a good way to surface genuinely novel tickets, but it has no native
`predict` for unseen points.

**Label** (`itsm/analysis/labeling.py`). A cluster id means nothing to a
service-desk manager. Each cluster gets:

* **c-TF-IDF terms** - documents of a cluster are concatenated into one pseudo
  document and weighted against the other clusters
  (`w(t,c) = tf(t,c) · log(1 + A/f(t))`), so the terms that surface are those
  frequent *inside* and rare *elsewhere*;
* **representative tickets** - the three designations closest to the centroid.

### Reading the metrics

| Metric | Reads as |
|--------|----------|
| `silhouette` | Geometric separation, in `[-1, 1]`. On short text it is a *relative* measure between runs, not an absolute grade; measured here at 0.15 (k=12) to 0.43 (k=36). It rises monotonically with k over the candidate range, so the range itself is the real prior - do not read the automatic selection as an optimum. |
| `davies_bouldin` | Lower is better; useful to compare `k` values. |
| `adjusted_rand`, `normalized_mutual_info` | Agreement with the declared taxonomy, chance-corrected. **Expect ARI to be low** (0.16 here): it penalises splitting one category into several pure clusters, which is precisely what happens and what we want to see. |
| `homogeneity` / `completeness` | Homogeneity = each cluster contains one category; completeness = each category lands in one cluster. The asymmetry tells you *which* of the two taxonomies is finer - measured 0.53 against 0.36, i.e. the clusters are finer than the routing taxonomy. |
| `category_purity` per cluster | The operational reading: 95% = the cluster is a category; 40% = a genuine mismatch worth investigating. Mean 0.66 at k=36. |

Full figures and the k-granularity trade-off: [`06-results.md`](06-results.md).

---

## 2.2 Ticket and status-update statistics

`itsm/analysis/lifecycle.py`. Two levels, because they answer different
questions.

### Ticket level

`kpi_overview` / `kpis_by(dimension)`:

* volume, tickets per day, **open backlog** and open ratio;
* **MTTR as median and p90, never as a mean** - handling times are
  heavy-tailed and a mean is dominated by a handful of month-long tickets;
* SLA breach rate, reopen rate, mean reassignments;
* **first-time-right rate** - share of tickets neither reassigned nor reopened,
  the single best summary of triage quality;
* updates per ticket.

Broken down by any column: `category`, `priority`, `assignment_group`, `site`,
`channel`.

### Flow level (the status updates)

The audit trail is the only place where *how work moves* is visible:

* **`state_transitions`** - the empirical transition matrix: counts,
  row-normalised probabilities and median dwell time before each transition.
  `Resolved -> In Progress` is the reopen loop; its probability is the
  quality-of-resolution indicator.
* **`dwell_times`** - median and p90 time spent in each state. A large
  `Awaiting User Info` p90 means the queue is waiting on users, not on
  engineers - a different problem from a capacity shortage.
* **`rework_profile`** - reassignment and reopen rates, share of tickets
  touched more than five times. This is the volume that routing automation
  targets.
* **`backlog_timeline`** - arrivals vs departures per period and the running
  backlog. Capacity is driven by the *difference*, not by the volume.
* **`aging_backlog`** - open tickets bucketed by age. A stable open count can
  hide a growing tail of old tickets; ageing shows it.
* **`followed_ticket`** - attributes plus the full chronology of one ticket,
  with the dwell time of each step. The view an analyst reads when auditing
  why a specific ticket took three weeks. Served by
  `GET /v1/tickets/{id}/trace`.

### Note on sources without an audit trail

The Hugging Face corpora have no events. The pipeline detects it, skips the
flow section and says so in the report rather than silently producing empty
tables.
