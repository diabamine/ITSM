"""Anonymisation pipeline: produce a ticket extract safe to analyse.

Runs before everything else. The raw extract is read once, the free-text
columns are rewritten, and the result is written next to it as
``tickets_anonymised.parquet``. Setting ``ITSM_USE_ANONYMISED=true`` then
points every other pipeline at that file without a single call site changing.

Two deliberate refusals:

* **The raw text is read from disk, never from** :func:`~itsm.data.loading.load_tickets`.
  That function honours ``use_anonymised``, so going through it would
  anonymise an already-anonymised extract and quietly report a second pass as
  a success.
* **Original text is not kept alongside** unless ``keep_original`` is set. An
  "anonymised" file that ships the raw column next to it protects nobody, so
  keeping it has to be an explicit, auditable choice.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from itsm.config import Settings, get_settings
from itsm.data.loading import anonymised_tickets_path, build_dataset
from itsm.data.schema import validate_tickets
from itsm.logging import get_logger
from itsm.pipelines.reporting import write_report
from itsm.privacy.anonymiser import TEXT_COLUMNS, Anonymiser, Backend

logger = get_logger("pipelines.anonymise")


def load_raw_tickets(settings: Settings | None = None) -> pd.DataFrame:
    """Read the un-anonymised extract, whatever ``use_anonymised`` says."""
    settings = settings or get_settings()
    tickets_path, _ = build_dataset(settings)
    return validate_tickets(pd.read_parquet(tickets_path))


def run(
    settings: Settings | None = None,
    backend: Backend | None = None,
    columns: tuple[str, ...] = TEXT_COLUMNS,
    keep_original: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """Anonymise the dataset and report what was replaced.

    Args:
        settings: Configuration; defaults to the process settings.
        backend: ``rules``, ``transformer`` or ``presidio``; defaults to the
            configured one.
        columns: Free-text columns to rewrite.
        keep_original: Also store the raw text as ``<column>_raw``.
        save: Write the anonymised extract to disk.

    Returns:
        The report, also written to ``reports/anonymisation.json``.
    """
    settings = settings or get_settings()
    backend = backend or settings.anonymisation_backend

    tickets = load_raw_tickets(settings)
    anonymiser = Anonymiser(
        backend=backend,
        model_name=settings.pii_model(backend),
        language=settings.presidio_language,
    )
    frame, outcomes = anonymiser.anonymise_frame(
        tickets, columns=columns, keep_original=keep_original
    )

    entity_counts: Counter[str] = Counter()
    tickets_touched = 0
    for outcome in outcomes.values():
        for spans in outcome.spans:
            for span in spans:
                entity_counts[str(span.entity)] += 1
    for index in range(len(frame)):
        if any(outcome.spans[index] for outcome in outcomes.values()):
            tickets_touched += 1

    if save:
        target = anonymised_tickets_path(settings)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        logger.info("wrote anonymised extract to %s", target)

    # Examples are drawn from the *output*, never the input: a report is a file
    # that gets copied around, and it must not become the PII dump the pipeline
    # exists to prevent.
    examples: list[dict[str, str]] = []
    for position in range(min(len(frame), 500)):
        if len(examples) >= 5:
            break
        if not any(outcome.spans[position] for outcome in outcomes.values()):
            continue
        row = frame.iloc[position]
        examples.append(
            {
                "designation": str(row["designation"]),
                "description": str(row["description"])[:220],
            }
        )

    report: dict[str, Any] = {
        "backend": backend,
        "dataset": settings.dataset,
        "n_tickets": len(frame),
        "columns": list(outcomes),
        "tickets_with_pii": tickets_touched,
        "tickets_with_pii_share": round(tickets_touched / max(len(frame), 1), 4),
        "spans_replaced": int(sum(outcome.n_spans for outcome in outcomes.values())),
        "spans_by_entity": dict(entity_counts.most_common()),
        "distinct_values_replaced": len(
            {
                value
                for outcome in outcomes.values()
                for mapping in outcome.mappings
                for value in mapping
            }
        ),
        "output": str(anonymised_tickets_path(settings)) if save else None,
        "anonymised_examples": examples,
    }
    write_report("anonymisation", report, settings)
    logger.info(
        "anonymised %d tickets with %s: %d spans replaced",
        len(frame),
        backend,
        report["spans_replaced"],
    )
    return report
