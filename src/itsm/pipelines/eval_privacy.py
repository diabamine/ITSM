"""Evaluation pipeline for anonymisation quality.

Four questions, in the order a data-protection officer and an ML engineer
actually ask them:

1. **Did we find the personal data?** Span-level recall at three strictnesses,
   per entity type, against the generator's ground truth.
2. **Is any of it still in the output?** The values are searched for in the
   rewritten text. This is the only number that cannot be gamed by generous
   span matching, and the one to put in front of a reviewer.
3. **What did it cost?** False positives, characters needlessly destroyed,
   and - the number nobody usually produces - the **downstream utility
   delta**: the category classifier is retrained on the anonymised text and
   compared with the same model trained on the original. A privacy control
   whose cost is unmeasured will be switched off the first time someone blames
   it for a bad model.
The pipeline scores each backend on the same tickets and the same split, so
``rules``, ``transformer`` and ``presidio`` are directly comparable.

Only a dataset carrying ``pii_spans`` can be scored - today that means the
generated one. On a real extract the same pipeline still runs the anonymiser
and reports volumes, but the quality section is replaced by an explicit "no
ground truth" note rather than a fabricated score; the honest path there is a
few hundred manually annotated tickets.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from itsm.config import Settings, get_settings
from itsm.logging import get_logger
from itsm.models.classifier import TicketClassifier
from itsm.pipelines.anonymise import load_raw_tickets
from itsm.pipelines.eval_classifier import evaluate as evaluate_classifier
from itsm.pipelines.eval_classifier import split_dataset
from itsm.pipelines.reporting import write_report
from itsm.privacy.anonymiser import Anonymiser, Backend
from itsm.privacy.entities import EntityType, PiiSpan
from itsm.privacy.evaluate import entity_distribution, residual_pii_rate, score_detection

logger = get_logger("pipelines.eval_privacy")


def _ground_truth(tickets: pd.DataFrame, column: str) -> list[list[PiiSpan]]:
    """Extract the reference spans of one column from the stored JSON."""
    references: list[list[PiiSpan]] = []
    for payload in tickets["pii_spans"]:
        if not payload:
            references.append([])
            continue
        spans = json.loads(payload).get(column, [])
        references.append([PiiSpan.from_dict(item) for item in spans])
    return references


def _strip_examples(report: dict[str, Any]) -> dict[str, Any]:
    """Remove verbatim PII from a report.

    Error analysis needs the actual strings, but a report file gets copied into
    tickets and chat. On generated data the values are invented and safe to
    show; on anything else they are replaced by counts.
    """
    cleaned = dict(report)
    for key in ("top_misses", "top_false_positives"):
        if key in cleaned:
            cleaned[key] = f"{len(cleaned[key])} distinct values (hidden: real data)"
    if "examples" in cleaned:
        cleaned["examples"] = f"{len(cleaned['examples'])} values (hidden: real data)"
    return cleaned


def _utility_delta(tickets: pd.DataFrame, redacted_designations: list[str]) -> dict[str, Any]:
    """Measure what anonymisation costs the downstream classifier.

    The same split, the same estimator, two versions of the input text: the
    original and the redacted one. A privacy control whose cost is unmeasured
    gets switched off the first time somebody blames it for a bad model, so the
    cost is a number here rather than an assumption.
    """
    frames = {
        "original": tickets,
        "redacted": tickets.assign(designation=redacted_designations),
    }
    scores: dict[str, float] = {}
    for name, frame in frames.items():
        x_train, x_test, y_train, y_test = split_dataset(frame, strategy="stratified")
        classifier = TicketClassifier(model="linear_svc").fit(x_train, y_train)
        metrics = evaluate_classifier(classifier, x_test, y_test)
        scores[name] = float(metrics["macro_f1"])
        logger.info("utility (%s): macro_f1=%.4f", name, scores[name])

    return {
        "macro_f1": {name: round(value, 4) for name, value in scores.items()},
        "cost_of_redaction": round(scores["original"] - scores["redacted"], 4),
    }


#: Names that appear in **neither** the generator pools nor the rule engine's
#: gazetteer, spanning naming traditions a French/English first-name list will
#: never cover. Invented, and deliberately kept out of `itsm.data.catalog`.
UNSEEN_NAMES: tuple[str, ...] = (
    "Krzysztof Nowak",
    "Oluwaseun Adeyemi",
    "Mei Ling Chen",
    "Ludivine Ferrand",
    "Anushka Rajagopalan",
    "Bjorn Gustafsson",
    "Ioana Petrescu",
    "Tarek Haddad",
    "Siobhan O'Donnell",
    "Yuki Nakamura",
    "Dimitrios Papadakis",
    "Aleksandra Kowalczyk",
)

#: Sentence frames the probe names are dropped into, mirroring the corpus.
_PROBE_FRAMES: tuple[str, ...] = (
    "Merci de contacter {name} au 06 12 34 56 78.",
    "Le poste de {name} ne demarre plus depuis ce matin.",
    "Dossier ouvert par {name} pour le compte de l'equipe achats.",
    "Please contact {name} before 5pm about the VPN issue.",
    "The mailbox of {name} has exceeded its quota.",
    "{name} cannot log in to SAP since the last update.",
)


def unseen_name_benchmark(
    settings: Settings | None = None,
    backends: tuple[Backend, ...] = ("rules",),
    isolate: bool = False,
) -> dict[str, Any]:
    """Score PERSON recall on names no component has ever seen.

    This exists because the headline comparison is rigged in the rule engine's
    favour and saying so is not enough: the generator samples names from
    ``itsm.data.catalog``, and the gazetteer is built from the same lists, so
    its PERSON recall is near-perfect *by construction*. A real corpus is full
    of names no list contains.

    The probe re-runs each backend on :data:`UNSEEN_NAMES` in corpus-like
    sentences. It is small and synthetic, so it proves a direction rather than
    a number - but the direction is the one that decides which backend to
    deploy.

    Args:
        settings: Configuration.
        backends: Backends to probe.
        isolate: Strip the rule layer from every model-based backend, so each
            one is measured on its own recognisers.

    Returns:
        Per-backend recall on the probe, with the names each one missed.
    """
    settings = settings or get_settings()
    probes = [(frame.format(name=name), name) for name in UNSEEN_NAMES for frame in _PROBE_FRAMES]
    texts = [text for text, _ in probes]

    results: dict[str, Any] = {
        "n_probes": len(probes),
        "n_names": len(UNSEEN_NAMES),
        "backends": {},
    }
    for backend in backends:
        anonymiser = Anonymiser(
            backend=backend,
            model_name=settings.pii_model(backend),
            language=settings.presidio_language,
            with_structural=False if isolate else None,
        )
        outcome = anonymiser.anonymise_column(texts)
        found = 0
        missed: Counter[str] = Counter()
        for (_, name), spans, anonymised in zip(
            probes, outcome.spans, outcome.anonymised, strict=True
        ):
            covered = any(
                span.entity is EntityType.PERSON and span.text and span.text in name
                for span in spans
            )
            # The end-to-end truth: is the name still readable in the output?
            if covered and name not in anonymised:
                found += 1
            else:
                missed[name] += 1

        results["backends"][backend] = {
            "recall": round(found / len(probes), 4),
            "missed_probes": sum(missed.values()),
            "names_never_caught": sorted(
                name for name in UNSEEN_NAMES if missed[name] == len(_PROBE_FRAMES)
            ),
            "names_partially_caught": sorted(
                name for name in UNSEEN_NAMES if 0 < missed[name] < len(_PROBE_FRAMES)
            ),
        }
    return results


def run(
    settings: Settings | None = None,
    backends: tuple[Backend, ...] = ("rules",),
    column: str = "description",
    sample_size: int | None = 3000,
    with_utility: bool = True,
    isolate: bool = False,
) -> dict[str, Any]:
    """Score one or more anonymisation backends.

    Args:
        settings: Configuration.
        backends: Backends to evaluate; add ``"transformer"`` when the ``hf``
            extra is installed, ``"presidio"`` when the ``presidio`` one is.
        column: Free-text column carrying the ground truth to score against.
        sample_size: Cap on tickets scored, for runtime.
        with_utility: Also retrain the classifier to measure the utility cost.
        isolate: Score every backend on its own recognisers, stripping the rule
            layer from ``transformer`` (``presidio`` already runs alone). This
            is the honest setting for *choosing* a detector: composed, a
            backend's gaps are hidden by the layer underneath. Leave it off to
            measure what a deployment would actually run.

    Returns:
        The report, also written to ``reports/privacy.json``.
    """
    settings = settings or get_settings()
    tickets = load_raw_tickets(settings)
    if sample_size and len(tickets) > sample_size:
        tickets = tickets.sample(sample_size, random_state=42).reset_index(drop=True)

    has_truth = "pii_spans" in tickets.columns and tickets["pii_spans"].notna().any()
    references = _ground_truth(tickets, column) if has_truth else []
    texts = tickets[column].fillna("").astype(str).tolist()
    show_examples = settings.dataset == "synthetic"

    report: dict[str, Any] = {
        "dataset": settings.dataset,
        "column": column,
        "n_tickets": len(tickets),
        "has_ground_truth": has_truth,
        "entity_distribution": entity_distribution(references) if has_truth else {},
        "backends": {},
    }
    if not has_truth:
        report["note"] = (
            "This dataset carries no PII annotation, so detection quality cannot be "
            "scored. Volumes are reported; for a real extract, annotate a few hundred "
            "tickets and rerun."
        )

    for backend in backends:
        anonymiser = Anonymiser(
            backend=backend,
            model_name=settings.pii_model(backend),
            language=settings.presidio_language,
            with_structural=False if isolate else None,
        )
        outcome = anonymiser.anonymise_column(texts)
        entry: dict[str, Any] = {"spans_detected": outcome.n_spans}

        if has_truth:
            detection = score_detection(references, outcome.spans, texts=texts)
            residual = residual_pii_rate(outcome.anonymised, references)
            entry["detection"] = detection if show_examples else _strip_examples(detection)
            entry["residual"] = residual if show_examples else _strip_examples(residual)

        if with_utility:
            designations = tickets["designation"].fillna("").astype(str).tolist()
            redacted = anonymiser.anonymise_column(designations).anonymised
            entry["utility"] = _utility_delta(tickets, redacted)

        report["backends"][backend] = entry

    report["isolated"] = isolate
    report["unseen_name_probe"] = unseen_name_benchmark(settings, backends, isolate)
    write_report("privacy", report, settings)
    return report
