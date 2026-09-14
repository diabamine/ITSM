# ADR 0002 - Anonymisation of ticket free text

* **Status**: accepted
* **Date**: 2026-09-14
* **Context**: ticket bodies carry personal data that no pipeline should read

## Context

Every model in this project reads `designation` and `description`. Those are
the two fields a requester types freely, and in a real service desk they
contain colleagues' names, mobile numbers, mailboxes, matricules and the
occasional pasted social-security number. The data was never deliberately
collected, it is personal data under GDPR, and it flows into training sets,
model artifacts, reports and API logs.

## Decisions

### 1. Anonymisation is a pipeline stage, not a utility function

`itsm privacy anonymise` writes `tickets_anonymised.parquet` next to the raw
extract, and `ITSM_USE_ANONYMISED=true` points every other pipeline at it
without a single call site changing.

*Rationale*: making it a stage means it can be **evaluated, versioned and
audited**. Making it a helper that each caller remembers to invoke means it is
skipped exactly once, in the script that mattered.

*Consequence*: `load_tickets()` fails loudly if the flag is set and the
anonymised extract is missing, rather than falling back to raw text. Failing
open on a privacy control is how leaks happen.

### 2. Three detectors, all shipped, scored against each other

A rule engine (offline, no download), a Hugging Face NER model (better on
unseen names) and Microsoft Presidio's analyser (the toolkit a security review
already knows), sharing one `Recognizer` interface.

*Rationale*: they fail in opposite directions. Formats are patterns, so rules
win on IBANs and mailboxes and always will. Names are language, so a model
wins on anyone whose name is not in a list. Picking one on principle would be
guessing; all three are measured on the same corpus.

*Consequence*: the transformer backend **includes** the rule engine's
structured layer rather than replacing it. There is no configuration in which
IBAN detection depends on a neural network.

*Amended*: the Presidio backend, added later, **does not** compose by default.
A composed detector cannot be benchmarked - its gaps are covered by the layer
underneath, so the score measures the union rather than the tool. Presidio
therefore runs alone, and `itsm privacy eval --isolate` strips the rule layer
from the transformer too so that a comparison holds every backend to the same
terms. The composed mode remains the deployment recommendation; the isolated
mode is how the choice between them is made.

*Risk accepted*: the headline benchmark flatters the rule engine, because the
generator's names come from the same list its gazetteer holds. Rather than
bury that, `unseen_name_benchmark()` measures the gap it creates (0.67 vs 0.94
recall), and the documentation leads with that number.

### 3. Replace every identifier with its entity tag

`<PERSON>`, `<EMAIL>`, `<IBAN>` - no realistic surrogates.

*Rationale*: a fake name that looks real is indistinguishable from a real name
to every downstream reader, so a leak and a correct redaction look identical.
That is the one property a privacy control cannot afford. A tag is
unmistakable and self-documenting.

*Consequence*: no keyed secret, no salt, no surrogate pools, no collision rate,
and no chance of a surrogate accidentally reproducing part of the value it
replaced - a real defect the earlier pseudonymising design exhibited on ~4% of
people.

*Risk accepted*: a tag carries no identity, so the text no longer links two
tickets from the same requester. Linkage must come from the structured caller
id column instead - where it belonged in the first place. `Policy.KEEP` remains
available per entity type for a deployment that judges one non-identifying.

### 4. Evaluation measures the output, not just the detector

Alongside span precision/recall, the pipeline searches the *rewritten text* for
the original values, and retrains the classifier on the anonymised corpus to
price the control.

*Rationale*: span metrics can be gamed by generous boundary matching, and a
control with an unmeasured cost gets switched off under pressure. The residual
check found a real defect that span metrics scored as a success: the earlier
pseudonymising design reusing the value's own first name in its surrogate.

### 5. Ground truth is generated, and its limits are stated

The synthetic generator plants identifiers with character offsets. No public
ITSM corpus is annotated for PII.

*Consequence*: absolute numbers transfer poorly to a real extract; the
*protocol* transfers. On real data the pipeline reports volumes and says
plainly that quality is unscored until a few hundred tickets are annotated.

## Alternatives considered

| Alternative | Rejected because |
|-------------|-------------------|
| Microsoft Presidio *as the only detector* | Rejected: it pulls spaCy plus its models as a hard dependency, and the point of the rule backend is that it needs *nothing*. **Superseded in part** - the architecture was already Presidio-shaped, so it was later added behind `Recognizer` as a third backend under the optional `presidio` extra, standalone and benchmarked against the other two. The rejection stands for making it mandatory; it does not stand for offering it. |
| An LLM for de-identification | Per-ticket cost and latency at service-desk volume, no offline path, non-deterministic output on a control that must be auditable, and the text would have to leave the perimeter to be scrubbed - which is the problem, not the solution. |
| Dropping the free-text columns entirely | Removes the personal data and the entire project with it: the designation is what every model reads. |
| Hashing each value into an opaque token (`<PERSON_a3f91>`) | Restores cross-ticket linkage, but reintroduces a keyed secret to manage and makes the text markedly harder to read for the agent triaging the ticket. Worth revisiting only if an analysis genuinely needs linkage the caller id cannot provide. |
| Anonymising at ingestion only, keeping raw text elsewhere | An "anonymised" extract sitting beside the raw one protects nobody. `keep_original` exists but is off by default and must be an explicit, auditable choice. |
