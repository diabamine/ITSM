"""Training pipeline for the category classifier.

One command, one reproducible artifact: load -> split -> fit -> evaluate ->
persist model + metrics. The evaluation is part of the training run, never an
afterthought: a model is only written to the store together with the scores it
obtained, so the API can always answer "how good is the model serving me?".
"""

from __future__ import annotations

from typing import Any, Literal

from itsm.config import Settings, get_settings
from itsm.data.loading import load_tickets
from itsm.logging import get_logger
from itsm.models.artifacts import ArtifactMetadata, ArtifactStore
from itsm.models.classifier import ModelName, TicketClassifier
from itsm.pipelines.eval_classifier import (
    SplitStrategy,
    compare_models,
    cross_validate,
    evaluate,
    split_dataset,
)
from itsm.pipelines.reporting import write_report

logger = get_logger("pipelines.train_classifier")

ARTIFACT_NAME = "classifier"


def run(
    settings: Settings | None = None,
    model: ModelName = "linear_svc",
    backend: Literal["tfidf", "embeddings"] | None = None,
    strategy: SplitStrategy = "temporal",
    text_column: str = "designation",
    label_column: str = "category",
    test_size: float = 0.2,
    compare_baselines: bool = True,
    with_cross_validation: bool = False,
    save: bool = True,
) -> dict[str, Any]:
    """Train, evaluate and persist the classifier.

    Args:
        settings: Configuration; defaults to the process settings.
        model: Estimator to train.
        backend: Text representation; defaults to the configured backend.
        strategy: ``temporal`` (default) or ``stratified`` split.
        text_column: Column used as model input.
        label_column: Column to predict.
        test_size: Share of tickets held out.
        compare_baselines: Also score the majority and Naive Bayes baselines.
        with_cross_validation: Add a 5-fold macro-F1 estimate (slower).
        save: Persist the fitted model to the artifact store.

    Returns:
        The evaluation report, also written to ``reports/classifier.json``.
    """
    settings = settings or get_settings()
    backend = backend or settings.embedding_backend

    tickets = load_tickets(settings)
    x_train, x_test, y_train, y_test = split_dataset(
        tickets,
        text_column=text_column,
        label_column=label_column,
        strategy=strategy,
        test_size=test_size,
    )
    logger.info("training on %d tickets, testing on %d", len(x_train), len(x_test))

    classifier = TicketClassifier(model=model, backend=backend).fit(x_train, y_train)
    metrics = evaluate(classifier, x_test, y_test)

    report: dict[str, Any] = {
        "model": model,
        "backend": backend,
        "split": strategy,
        "dataset": settings.dataset,
        "n_train": len(x_train),
        "label_column": label_column,
        "metrics": metrics,
    }
    if compare_baselines:
        report["model_comparison"] = compare_models(x_train, x_test, y_train, y_test).to_dict(
            orient="records"
        )
    if with_cross_validation:
        report["cross_validation"] = cross_validate(x_train, y_train, model=model)

    if save:
        ArtifactStore(settings.artifacts_dir).save(
            ARTIFACT_NAME,
            classifier,
            ArtifactMetadata(
                name=ARTIFACT_NAME,
                kind="classification",
                dataset=settings.dataset,
                n_train=len(x_train),
                params={"model": model, "backend": backend, "split": strategy},
                metrics={
                    key: metrics[key]
                    for key in ("macro_f1", "accuracy", "balanced_accuracy", "top3_accuracy")
                },
                notes=f"Predicts {label_column} from {text_column}.",
            ),
        )

    write_report("classifier", report, settings)
    logger.info("macro_f1=%.4f accuracy=%.4f", metrics["macro_f1"], metrics["accuracy"])
    return report
