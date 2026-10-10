# 8. Sub-topic tagging and root-cause attribution

```bash
uv run itsm analyze root-cause --backend embeddings
uv run itsm analyze root-cause --recent 30 --baseline 90 --labeller local
```

The weak-signal detector ([`03-modeling.md`](03-modeling.md)) answers *when* and
*in which category* volume moved. That is rarely the question anyone actually
has. "Network tickets are up 40%" cannot be acted on; *"78% of that increase is
one symptom - the VPN client crashing after the Windows update"* can.

The gap is granularity. A declared category is a **routing** taxonomy: a dozen
buckets chosen years ago so tickets reach the right queue. A root cause lives
one level below it, as a *sub-topic* nobody has labelled - because a root cause
is only known after it appears.

---

## 8.1 The shape of the problem

Three properties make this different from the clustering already in the project
([`02-analysis.md`](02-analysis.md)), and each one drives a design decision.

| Property | Consequence |
|---|---|
| Sub-topics are **not known in advance** | No supervised model. Discovery has to be unsupervised, and the taxonomy is an output, not an input. |
| The same failure is **worded differently** by every requester, and here in two languages | Keyword clustering splits one root cause into several. Embeddings are not a nicety; they are the requirement. |
| A tag is only useful if it **means the same thing next month** | Cluster ids must be stable across runs, or the trend you tracked in March points at something else in April. |

## 8.2 Pipeline

```
combined text -> embed once -> cluster within each category -> merge near-duplicates
              -> name -> stable ids -> attribute the increase
```

**Embed once, for the whole corpus.** Every centroid has to live in one vector
space. Encoding each category separately fits a different TF-IDF basis per
slice, and two centroids then have no comparable meaning - it also breaks
tagging of new tickets outright. The *fitted* encoder is kept on the model, so
an incoming ticket is projected into the space the topics were built in rather
than into a fresh one.

**Cluster within each category.** HDBSCAN where available: it finds groups of
uneven density and, crucially, refuses to assign outliers. Roughly 40% of
tickets end up unassigned on the reference corpus, and that is the correct
behaviour - a diffuse one-off is not a sub-topic, and forcing it into the
nearest one is how a real emerging failure gets buried.

**Merge near-duplicates.** Clustering separates the same symptom written in two
languages. Measured on the reference corpus: the French and English variants of
one Outlook failure sit at cosine **0.85**, a mailbox-quota pair at **0.73**,
while unrelated sub-topics of the same category reach only **0.61** at the 95th
percentile. A single agglomerative pass at **0.70** closes the gap with room to
spare.

This is not tidiness. Unmerged, the injected root cause below is reported as two
rows of 33% and 24%; merged, it is one row of **57%**. That is the difference
between spotting the driver and dismissing two minor ones.

**Name it.** Three interchangeable strategies behind one method
(`itsm/analysis/topic_labeller.py`):

| Labeller | What it does | When |
|---|---|---|
| `template` *(default)* | Uses the shortest representative ticket title | Offline, free, and hard to beat: a title a user wrote already *is* a description of the symptom |
| `local` | A small instruct model through `transformers` | Generalises the examples instead of copying one |
| `api` | Any OpenAI-compatible endpoint - vLLM, Ollama, Azure, an internal gateway | When the only reachable model is one somebody hosts internally |

An LLM failure is caught and falls back to the offline label. A naming step must
never be able to fail a pipeline: an unnamed cluster is cosmetic, a crashed
analysis is not.

**Keep ids stable.** A new run matches each centroid against the stored
taxonomy; above cosine 0.80 the topic inherits its id, below it a new id is
minted. Persisted as the `subtopics` artifact.

## 8.3 Attribution: the part that names a root cause

Running an anomaly detector per sub-topic does not work - the counts are small,
the series are noisy, and a detector answers yes/no where the question is *how
much of the rise is this one*.

So the increase is **decomposed** instead. For each sub-topic, over a recent
window and the baseline window immediately before it:

```
delta_per_day      = recent_rate - baseline_rate
share_of_increase  = delta_per_day / (sum of all positive deltas)
poisson_p          = P(>= n_recent | the rate had not changed)
```

Two rankings, which disagree on purpose:

* **Absolute contribution** ranks *what to fix*. A sub-topic going from 20 to 30
  tickets a day contributes ten times more than one going from 1 to 3, even
  though the second grew three-fold.
* **Growth ratio** ranks *what is new*. A sub-topic absent from the baseline has
  no ratio at all; those are reported separately as `emerging`, and they are the
  genuinely novel failure modes.

A row is promoted to **driver** only when it is both large (≥5% of the increase)
and unlikely to be noise (p ≤ 0.01). The p-value alone would promote a jump from
1 to 5 tickets; the share alone would promote a big family drifting upwards with
the business.

`share_of_increase` is a share of the **total** rise, including sub-topics too
small to rank, so the displayed column sums to less than 1. Rescaling it to the
visible rows would inflate every headline figure, which is the opposite of what
an attribution is for.

## 8.4 Does it work?

The generator now concentrates an injected surge on **one symptom template**
rather than lifting a whole family evenly - which is how a real root cause
behaves, and what makes the question measurable at all. The Email &
Collaboration ramp targets symptom #0, the Outlook connection failure.

Asked to explain the last 14 days of that ramp, with no knowledge of any of it:

| category | sub-topic | per day | was | × | share | p |
|---|---|---|---|---|---|---|
| **Email & Collaboration** | **Outlook bloque sur tentative de connexion depuis ce matin** | **3.36** | **0.38** | **8.9** | **57%** | **~0** |
| Network & VPN | Reseau tres lent sur le site de Brest | 0.50 | 0.21 | 2.3 | 5% | 3.4e-02 |
| Network & VPN | Le client VPN renvoie l'erreur 0xC004F074 au demarrage | 0.50 | 0.30 | 1.7 | 4% | 1.4e-01 |
| Security | L'antivirus a bloque un fichier, analyse necessaire | 0.36 | 0.18 | 2.0 | 3% | 1.1e-01 |

One sub-topic, correctly named, carrying **57% of the entire increase** at
p ≈ 0 - and everything below it at ≤5% with p > 0.03, i.e. noise. The injected
cause is recovered without being told a category, a date or a keyword.

## 8.5 Applying it to a real extract

```bash
ITSM_DATASET=csv ITSM_CSV_PATH=/path/to/export.csv \
  uv run itsm analyze root-cause --backend embeddings --recent 30 --baseline 90
```

Four things to check before trusting the output on your own data:

1. **Re-measure the merge threshold.** 0.70 is calibrated on this corpus. On
   yours, take the pairwise centroid similarities, look at where same-symptom
   pairs sit versus unrelated ones, and set the value in that gap. If there is
   no gap, the embedding model is not separating your vocabulary.
2. **Choose the windows from the incident, not the default.** `recent` should be
   about as long as the phenomenon you are chasing. A 14-day window will not
   explain a surge that started three months ago.
3. **Watch the coverage.** Much below 50% and `min_cluster_size` is too high for
   your volume; near 100% and HDBSCAN is not refusing anything, which usually
   means the clusters are not real.
4. **Read the unassigned pile.** It is where a brand-new failure mode sits
   before it is large enough to form a cluster.

## 8.6 Limits

* **Attribution is not causation.** This finds what grew, precisely and with a
  significance test. Why it grew - a release, a migration, a supplier - still
  comes from a human joining it to the change calendar.
* **A gradual, corpus-wide drift has no single driver** and will correctly show
  a flat ranking. That is an answer, not a failure.
* **Quality depends on the embedding model.** The default multilingual
  MiniLM is small and fast; a domain-adapted or larger model separates
  near-neighbour symptoms better, and the threshold should be re-measured after
  any model change.
* **The label is a summary, not a diagnosis.** Even with an LLM it describes the
  symptom the tickets share, not the fault underneath it.
