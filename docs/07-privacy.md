# 7. Anonymisation

```bash
uv run itsm privacy anonymise                      # produce a safe extract
uv run itsm privacy eval --backend rules --backend transformer
ITSM_USE_ANONYMISED=true uv run itsm train classifier   # analyse the safe copy
```

Ticket free text is the one field in an ITSM extract nobody controls. The
category is a dropdown, the priority is a matrix, but the body is whatever the
requester typed - and that routinely includes a colleague's name, a mobile
number for a callback, a mailbox, a matricule, and occasionally a
social-security number pasted by mistake. None of it was collected on purpose,
all of it is personal data, and it sits in the exact column every model in this
project reads.

This is therefore **step zero**, not a feature: `itsm.privacy` removes personal
data before any analysis runs, offers two interchangeable detectors, and -
the part that is usually missing - **measures** the result.

---

## 7.1 What counts as personal data here

Ten entity types, chosen because they actually turn up in ticket bodies rather
than to fill a taxonomy (`itsm/privacy/entities.py`):

| Entity | Example | Direct identifier |
|---|---|---|
| `PERSON` | `Camille Dubois` | ● |
| `EMAIL` | `camille.dubois@groupe-exemple.example` | ● |
| `PHONE` | `06 12 34 56 78`, `+44 7123 456789` | ● |
| `NATIONAL_ID` | French social-security number | ● |
| `IBAN` | `FR76 3000 1007 9412 …` | ● |
| `CREDIT_CARD` | 16 grouped digits | ● |
| `EMPLOYEE_ID` | `EMP-45231`, `matricule 402118` | |
| `ADDRESS` | `12 rue de la Paix, 75002 Paris` | |
| `DATE_OF_BIRTH` | `14/07/1988` | |
| `IP_ADDRESS` | `10.20.30.40` | |

The **direct identifiers** flag matters: one such value names a person on its
own, so missing one is a breach rather than a quality issue. The evaluation
reports their leak rate separately for exactly that reason.

## 7.2 Ground truth: the dataset knows where the PII is

Anonymisation cannot be scored without knowing where the personal data was, and
no public ITSM corpus is annotated for it. The generator therefore plants the
identifiers itself and records the **character offsets** of every one
(`itsm/data/pii.py`), stored per ticket in the `pii_spans` column as
`{"designation": [...], "description": [...]}`.

Two rules keep that reference trustworthy:

* **The value is sampled first and the text is built around it.** Offsets are
  accumulated while concatenating, never recovered afterwards by searching for
  the value - a search would mislabel the second occurrence of a common first
  name and silently corrupt the reference. A test asserts that every stored
  offset slices exactly the stored value.
* **Every value is invented**, mail domains sit in the RFC 2606 reserved
  `.example` space, and the IBAN and card number match the right *shape* while
  deliberately failing their checksum - a leaked sample cannot be mistaken for
  a usable credential.

On the reference dataset this yields **53% of tickets carrying at least one
identifier**, 9 448 spans over 7 897 tickets, every entity type represented.
Sentences are separated by a full stop, which sounds trivial and is not: joined
with a bare space, `Mailbox transfer for Hannah Parker Workstation of Andrew
Evans` is ambiguous to a human, and scoring a detector against an ambiguity the
generator invented measures nothing.

## 7.3 Three ways to find it

### A. Rule engine - offline, no download (`itsm/privacy/rules.py`)

Regular expressions, contextual triggers and an authored gazetteer. Nothing is
downloaded, imported from a model hub or trained. That is a **capability**, not
a fallback: in a constrained environment a detector needing no network, no GPU
and no model licence is often the only deployable one.

Two layers that fail differently:

* **Structured identifiers** (email, phone, IBAN, card, national ID, employee
  number, IP, postal address) are matched by *shape*. Precision is essentially
  1.0 - a string matching the IBAN pattern is an IBAN - and recall is bounded
  only by formats nobody anticipated.
* **Person names** have no pattern, so three weaker signals are combined:
  contextual triggers (`contacter`, `de la part de`, `raised by`, `M.`/`Mme`)
  followed by a capitalised sequence; a gazetteer of first names; and a
  deliberate refusal to flag bare capitalised sequences, which in ticket text
  are overwhelmingly products and sites (`Outlook`, `Paris`, `SAP`).

Two defects worth recording, both found by measuring rather than reading:

* the trigger pattern needs a **left word boundary** - without it `telephone`
  ends in the trigger `phone` and `5pm.` ends in the trigger `M.`, so every
  following sentence started with a "person";
* `phone` had to be dropped as a trigger entirely: in ITSM text it is a device
  nine times out of ten.

### B. Hugging Face NER - reads the sentence (`itsm/privacy/transformer.py`)

`Davlan/distilbert-base-multilingual-cased-ner-hrl` by default: small,
CPU-friendly, covers both languages of the corpus. For a French-only deployment
`Jean-Baptiste/camembert-ner` is usually stronger (`ITSM_NER_MODEL`).

**It is not used alone.** A NER model is trained on PER/ORG/LOC and knows
nothing about IBANs or matricules, so this backend keeps the rule engine's
structured layer for formats and uses the model for names. The hybrid is the
honest engineering answer, and the per-entity numbers below show why.

`ORG` predictions are deliberately dropped: in ticket text the organisations
are the employer and its vendors (`SAP`, `Microsoft`), which are not personal
data and whose redaction would destroy the vocabulary the classifier lives on.

### C. Presidio - the standard analyser (`itsm/privacy/presidio.py`)

```bash
ITSM_ANONYMISATION_BACKEND=presidio uv run itsm privacy anonymise
```

Presidio and its spaCy pipeline are mandatory dependencies; the pipeline
(`en_core_web_md` by default, `fr_core_news_md` for French) is downloaded on
first use. `md` rather than `lg` costs a little `PERSON` recall for a tenth of
the download - set `ITSM_PRESIDIO_SPACY_MODEL=en_core_web_lg` and re-run the
evaluation to price that.

Microsoft Presidio's `AnalyzerEngine` runs a spaCy pipeline for the entities
that need language understanding and pattern recognisers for the ones that have
a shape. The reason to offer it as a third option is not accuracy in the
abstract - it is that **it is the toolkit a security review already knows**.
The rule engine is authored for this corpus and the transformer backend is one
checkpoint; Presidio is maintained, audited and extended outside this project,
and in an organisation that has standardised on it, "we use Presidio" ends a
conversation that "we wrote regexes" starts.

Two things it genuinely does better than either existing backend:

* **Validated structured identifiers.** `IBAN_CODE` and `CREDIT_CARD` are
  checksum-verified rather than shape-matched, so a 16-digit order number that
  the regex layer here accepts is rejected.
* **A context layer**, which raises a match's score when nearby words support
  it - the rule engine's trigger idea, generalised and maintained upstream.

**This backend runs Presidio alone**, and that is the difference from backend
B. The transformer backend composes itself with the rule engine's structured
layer; this one does not, because **a composed detector cannot be
benchmarked**. Every number a composed backend produces is the union of two
systems: a gap in one is invisible while the other covers it, and the
comparison measures the union rather than the tool. Isolated, `presidio`
answers the question the benchmark is actually asking - *what does this find on
its own?*

The consequence is a finding, not a defect to paper over: Presidio's predefined
recognisers have no concept of a French *matricule*, a NIR, or the
street-address shapes of this corpus, so standalone it misses them. That is
asserted in the test suite rather than left to be discovered in production.
`with_structural=True` composes the rule layer back in - for a deployment that
has already chosen Presidio, not while the choice is being made.

The same entities are dropped as in backend B, plus `DATE_TIME`: a bare date is
not personal data, only a date introduced by a birth trigger is - and with no
rule layer underneath, standalone Presidio does not detect dates of birth at
all.

### Benchmarking the three on the same terms

```bash
uv run itsm privacy eval --backend rules --backend transformer --backend presidio --isolate
```

`--isolate` strips the rule layer from `transformer` too, so all three are
scored on their own recognisers. Without it, `rules` and `presidio` run alone
while `transformer` runs composed, and the table compares two detectors with a
hybrid. Both readings are legitimate and they answer different questions:

| | what it measures | use it to |
|---|---|---|
| `--isolate` | each tool on its own | **choose** a detector |
| default | what a deployment would run | **size** the residual risk |

The report records which mode produced it under `isolated`.

The same comparison, cell by cell and with the per-entity gaps, the error
analysis and the throughput cost laid out side by side:
[`notebooks/anonymisation_backend_comparison.ipynb`](../notebooks/anonymisation_backend_comparison.ipynb).
It runs on a sample (500 tickets by default) and skips any backend whose extra
is not installed, so it works with `rules` alone.

**The cost is one engine, one language.** An `AnalyzerEngine` is built around a
single spaCy pipeline, and nothing in a ticket declares which language it is
in. On this bilingual corpus that is a real trade rather than a configuration
detail: French text analysed by an English pipeline loses the French-only names
that no gazetteer entry rescues. `ITSM_PRESIDIO_LANGUAGE` and
`ITSM_PRESIDIO_SPACY_MODEL` set it; running the evaluation under both is how
you find out what it costs on your own data.

> **Not yet scored on this corpus.** The numbers in §7.6 cover `rules` and
> `transformer` only, and they were produced in the default (composed) mode.
> Presidio is wired into the same evaluation and produces the same table and
> the same unseen-name probe, but no run is recorded here yet, and inventing
> one would defeat the point of the section. Run it before choosing it.

## 7.4 Replacement: the entity tag, nothing else

Every detected value becomes the name of its kind:

```
Merci de contacter Camille Dubois au 06 12 34 56 78 ou
camille.dubois@groupe-exemple.example, matricule EMP-45231, nee le 14/07/1988.
->
Merci de contacter <PERSON> au <PHONE> ou <EMAIL>, matricule <EMPLOYEE_ID>,
nee le <DATE_OF_BIRTH>.
```

**Why a tag rather than a realistic fake value.** A surrogate that looks like a
real name is indistinguishable from a real name to everyone downstream - the
agent reading the ticket, the analyst reading an export, the reviewer auditing
the output. A leak and a correct redaction look identical, which is exactly the
property a privacy control must not have. `<PERSON>` is unmistakable and
self-documenting: you can see at a glance that the field was processed, and
what was taken out of it.

It also removes a whole class of failure. An earlier design pseudonymised names
with keyed surrogates drawn from the same name pools the corpus uses, and ~4%
of the time the digest landed on a name the source already contained -
`Rachel Parker` became `Rachel Benali`, leaving the real first name in the
output. Tags cannot do that, and the keyed secret, the salt, the collision rate
and the surrogate-reuse guard all disappear with them.

**What it costs, stated plainly.** A tag carries no identity, so two tickets
from the same requester are no longer linkable *through the text*. If an
analysis needs that linkage it has to come from a structured key - the caller
id column - which is where it should have come from anyway. Per-entity policy
remains configurable: `Policy.KEEP` exempts a type a deployment judges
non-identifying, and that decision is data rather than code.

## 7.5 Measuring it - four questions

`itsm/privacy/evaluate.py`, run by `itsm privacy eval`.

**1. Did we find it?** Three levels of recall, because they mean different
things:

* `recall_strict` - exact boundaries and right type. The academic number.
* `recall_partial` - overlapping enough (IoU ≥ 0.5) and right type.
* `recall_coverage` - touched at all, any type. **The privacy-relevant one**:
  mislabelling an email as a person still removes it.

**2. Is any of it still in the output?** The values are searched for in the
*rewritten text*. This is the only number span arithmetic cannot flatter, and
it comes in two flavours: whole-value survival, and `residual_partial_rate`
for the classic sloppy-boundary failure where `Jean Dupont` is redacted from
the surname onwards and `Jean` stays behind.

**3. What did it cost?** False positives, characters needlessly destroyed, and
the number nobody usually produces: the **downstream utility delta**. The
category classifier is retrained on the anonymised text and compared with the
same model trained on the original. A privacy control whose cost is unmeasured
gets switched off the first time somebody blames it for a bad model.

## 7.6 Results

Measured on 2 000 tickets of the reference dataset, 2 087 annotated values.

| | rules | transformer |
|---|---|---|
| Coverage recall | **1.000** | 0.999 |
| Exact-boundary recall | **1.000** | 0.996 |
| Precision | **0.999** | 0.976 |
| F1 | **0.9995** | 0.987 |
| Leak rate, direct identifiers | **0.000** | 0.0012 |
| **Residual PII in the output** | **0 / 2 087** | 2 / 2 087 |
| False-positive spans | **2** | 50 |
| Characters over-redacted | 0.01% | 0.13% |
| Utility cost (macro-F1) | **0.000** | −0.001 |

Per entity, the rule engine scores recall **1.00 on all ten types**. The
transformer matches it everywhere except `PERSON` (recall 0.997, precision
0.951): its false positives are sentence-initial capitalised verbs - `Cannot`
30 times, `Can` 9 times - which the rule engine's capitalisation-plus-context
requirement rejects by construction.

### The number that actually decides which one to deploy

**The table above is rigged in the rule engine's favour, and saying so is not
enough.** The generator samples names from `itsm/data/catalog.py`, and the
gazetteer is built from the same lists: its `PERSON` recall is near-perfect *by
construction*. A real corpus is full of names no list contains.

So `unseen_name_benchmark()` re-runs both backends on twelve invented names
spanning naming traditions a French/English list will never cover - Krzysztof
Nowak, Oluwaseun Adeyemi, Mei Ling Chen, Yuki Nakamura - in corpus-like
sentences:

| Backend | Recall on unseen names |
|---|---|
| rules | **0.667** |
| transformer | **0.944** |
| both, on the same names *with* a trigger word | 1.000 |

The rule engine still catches every name that follows a trigger
(`ouvert par Krzysztof Nowak`); it misses the ones that do not
(`Krzysztof Nowak cannot log in to SAP`), because neither of its two signals
fires. That is a 33% miss rate on personal data in the realistic case.

**Recommendation: deploy the transformer where names matter and the `hf` extra
is installable; keep the rule engine as the offline path and for every
structured identifier.** They are not competitors - the transformer backend
already runs the rules underneath.

### Utility cost: zero

Redacting the designation costs the classifier **nothing** (0.0000 macro-F1).
Names carry no category signal, rare name tokens are dropped by the TF-IDF
minimum document frequency, and `<PERSON>` is if anything a mild signal for the
families whose titles name people. The caveat is worth stating: this is measured on generated data where
names are uninformative by construction. On a real corpus the risk is
*over*-redaction eating product vocabulary - which is precisely what
`over_redacted_char_share` and this delta exist to catch.

## 7.7 Known limitations

* **All-caps text defeats the capitalisation heuristics.** A shouted title
  yields the one rule-engine false positive on the whole corpus
  (`THE PRINTER`). Over-redaction is the safe direction, so it stands.
* **The gazetteer is the deployment's own directory.** Extending it with real
  first names is legitimate and far more effective than any generic list - and
  it is the single cheapest improvement to the offline path.
* **Ground truth exists only on generated data.** On a real extract the
  pipeline still anonymises and reports volumes, but the quality section is
  replaced by an explicit "no ground truth" note rather than a fabricated
  score. The honest path there is a few hundred manually annotated tickets;
  the evaluation code takes them unchanged.
* **Entity tags break cross-ticket linkage through the text.** Counting how
  many tickets one requester opened has to use the structured caller id, not
  the body. That is the deliberate trade for an output in which a leak is
  visible at a glance.
* **Redacting the text is not GDPR anonymisation of the dataset.** The
  structured columns remain: site plus opening date plus category can still
  single somebody out in a small team. Closing that needs k-anonymity work on
  those columns, which is out of scope here and would be the next step before
  any external sharing.
