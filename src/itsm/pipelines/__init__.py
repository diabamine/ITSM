"""End-to-end pipelines: analysis, training and evaluation.

Each module exposes a single ``run()`` returning a JSON-serialisable report and
persisting it under ``reports/``. The CLI and the tests call nothing else.
"""

from itsm.pipelines import (
    anonymise,
    eval_privacy,
    eval_sentiment,
    eval_weak_signal,
    run_analysis,
    train_classifier,
)
from itsm.pipelines.reporting import read_report, write_report

__all__ = [
    "anonymise",
    "eval_privacy",
    "eval_sentiment",
    "eval_weak_signal",
    "read_report",
    "run_analysis",
    "train_classifier",
    "write_report",
]
