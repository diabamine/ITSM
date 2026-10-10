"""Command line interface.

The CLI is a thin shell over :mod:`itsm.pipelines`: every command loads
settings, calls one ``run()`` and prints the report. No analysis logic lives
here, so the CLI, the API and the tests can never diverge.

Run ``itsm --help`` for the command tree.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from itsm.config import get_settings
from itsm.logging import setup_logging

app = typer.Typer(
    help="ITSM ticket analytics: analysis, model training and serving.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Build and inspect the dataset.", no_args_is_help=True)
analyze_app = typer.Typer(help="Descriptive analysis.", no_args_is_help=True)
train_app = typer.Typer(help="Model training.", no_args_is_help=True)
eval_app = typer.Typer(help="Model evaluation.", no_args_is_help=True)
privacy_app = typer.Typer(help="Anonymisation of personal data.", no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(analyze_app, name="analyze")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(privacy_app, name="privacy")

console = Console()


def _print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Render a flat metrics dictionary as a table."""
    table = Table(title=title, show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            table.add_row(key, str(value))
    console.print(table)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Configure logging for every command."""
    setup_logging("DEBUG" if verbose else get_settings().log_level)


# --------------------------------------------------------------------- data
@data_app.command("build")
def data_build(
    force: Annotated[bool, typer.Option(help="Rebuild even if cached.")] = False,
) -> None:
    """Materialise the configured dataset to Parquet."""
    from itsm.data.loading import build_dataset

    tickets_path, events_path = build_dataset(get_settings(), force=force)
    console.print(f"[green]tickets[/green] {tickets_path}")
    console.print(f"[green]events [/green] {events_path}")


@data_app.command("info")
def data_info() -> None:
    """Print the shape and the main distributions of the dataset."""
    from itsm.data.loading import load_events, load_tickets

    settings = get_settings()
    tickets, events = load_tickets(settings), load_events(settings)
    _print_metrics(
        f"dataset '{settings.dataset}'",
        {
            "tickets": len(tickets),
            "events": len(events),
            "categories": tickets["category"].nunique(),
            "from": str(tickets["opened_at"].min()),
            "to": str(tickets["opened_at"].max()),
            "languages": ", ".join(sorted(tickets["language"].dropna().unique())),
        },
    )
    console.print(tickets["category"].value_counts().to_string())


# ------------------------------------------------------------------ analyze
@analyze_app.command("designations")
def analyze_designations(
    k: Annotated[int | None, typer.Option(help="Number of families; omit to auto-select.")] = None,
    backend: Annotated[str | None, typer.Option(help="tfidf | embeddings")] = None,
    algorithm: Annotated[str, typer.Option(help="kmeans | hdbscan")] = "kmeans",
) -> None:
    """Cluster the designations into families and describe them."""
    from itsm.pipelines.run_analysis import run_designation_analysis

    report = run_designation_analysis(
        get_settings(),
        n_clusters=k,
        backend=backend,  # type: ignore[arg-type]
        algorithm=algorithm,  # type: ignore[arg-type]
    )
    _print_metrics("clustering quality", report["metrics"])
    table = Table(title="discovered families", header_style="bold")
    for column in ("id", "size", "share", "label", "dominant category", "purity"):
        table.add_column(column)
    for cluster in sorted(report["clusters"], key=lambda c: -c["size"]):
        table.add_row(
            str(cluster["cluster_id"]),
            str(cluster["size"]),
            f"{cluster['share']:.1%}",
            cluster["name"],
            str(cluster["dominant_category"]),
            f"{(cluster['category_purity'] or 0):.0%}",
        )
    console.print(table)


@analyze_app.command("lifecycle")
def analyze_lifecycle() -> None:
    """Compute ticket and status-update statistics."""
    from itsm.pipelines.run_analysis import run_lifecycle_analysis

    report = run_lifecycle_analysis(get_settings())
    _print_metrics("overview", report["overview"])
    if report.get("status_updates"):
        _print_metrics("rework", report["status_updates"]["rework"])


@analyze_app.command("root-cause")
def analyze_root_cause(
    recent: Annotated[int, typer.Option(help="Days of the window to explain.")] = 14,
    baseline: Annotated[int, typer.Option(help="Days of the comparison window.")] = 56,
    backend: Annotated[str | None, typer.Option(help="tfidf | embeddings")] = None,
    within: Annotated[str, typer.Option(help="Parent column to sub-divide.")] = "category",
    min_cluster_size: Annotated[int, typer.Option(help="Smallest sub-topic.")] = 20,
    labeller: Annotated[str, typer.Option(help="template | local | api")] = "template",
    llm_model: Annotated[str | None, typer.Option(help="Model id for the LLM labeller.")] = None,
    llm_url: Annotated[str | None, typer.Option(help="OpenAI-compatible base URL.")] = None,
) -> None:
    """Tag tickets with sub-topics and rank what is driving the increase."""
    from itsm.pipelines.root_cause import run

    options: dict[str, Any] = {}
    if labeller == "local" and llm_model:
        options["model_name"] = llm_model
    if labeller == "api":
        options = {"base_url": llm_url or "", "model": llm_model or ""}

    report = run(
        get_settings(),
        backend=backend,  # type: ignore[arg-type]
        within=within,
        recent_days=recent,
        baseline_days=baseline,
        min_cluster_size=min_cluster_size,
        labeller_kind=labeller,
        labeller_options=options,
    )
    _print_metrics(
        "sub-topics",
        {
            "tickets": report["n_tickets"],
            "sub-topics": report["n_subtopics"],
            "coverage": f"{report['coverage']:.0%}",
            "labeller": report["labeller"],
            "significant drivers": report["n_drivers"],
        },
    )

    if not report["ranking"]:
        console.print("[yellow]no sub-topic carried enough recent volume to rank[/yellow]")
        return

    table = Table(
        title=f"what changed over the last {report['recent_days']} days", header_style="bold"
    )
    for column in ("category", "sub-topic", "per day", "was", "x", "share", "p"):
        table.add_column(column)
    for row in report["ranking"][:10]:
        driver = row["poisson_p"] <= 0.01 and row["share_of_increase"] >= 0.05
        style = "bold red" if driver else ""
        table.add_row(
            row["parent_category"][:22],
            row["label"][:52],
            f"{row['recent_per_day']:.2f}",
            f"{row['baseline_per_day']:.2f}",
            f"{row['growth_ratio']:.1f}" if row["growth_ratio"] else "new",
            f"{row['share_of_increase']:.0%}",
            f"{row['poisson_p']:.1e}",
            style=style,
        )
    console.print(table)
    console.print("[dim]bold red = statistically significant driver of the increase[/dim]")


# -------------------------------------------------------------------- train
@train_app.command("classifier")
def train_classifier_command(
    model: Annotated[
        str, typer.Option(help="linear_svc | logreg | complement_nb | majority")
    ] = "linear_svc",
    backend: Annotated[str | None, typer.Option(help="tfidf | embeddings")] = None,
    split: Annotated[str, typer.Option(help="temporal | stratified")] = "temporal",
    label: Annotated[str, typer.Option(help="Column to predict.")] = "category",
    cross_validation: Annotated[bool, typer.Option("--cv", help="Add a 5-fold estimate.")] = False,
) -> None:
    """Train, evaluate and store the category classifier."""
    from itsm.pipelines.train_classifier import run

    report = run(
        get_settings(),
        model=model,  # type: ignore[arg-type]
        backend=backend,  # type: ignore[arg-type]
        strategy=split,  # type: ignore[arg-type]
        label_column=label,
        with_cross_validation=cross_validation,
    )
    metrics = report["metrics"]
    _print_metrics(
        f"classifier ({model}, {report['backend']}, {split} split)",
        {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
    )
    if "model_comparison" in report:
        table = Table(title="model comparison", header_style="bold")
        for column in ("model", "macro_f1", "accuracy", "top3_accuracy"):
            table.add_column(column)
        for row in report["model_comparison"]:
            table.add_row(
                row["model"],
                f"{row['macro_f1']:.4f}",
                f"{row['accuracy']:.4f}",
                f"{row['top3_accuracy']:.4f}",
            )
        console.print(table)


# --------------------------------------------------------------------- eval
@eval_app.command("sentiment")
def eval_sentiment_command(
    backend: Annotated[
        list[str] | None, typer.Option(help="Repeatable: lexicon, transformer.")
    ] = None,
    label_source: Annotated[str, typer.Option(help="generator | operational")] = "generator",
) -> None:
    """Score the sentiment backends against the weak labels."""
    from itsm.pipelines.eval_sentiment import run

    report = run(
        get_settings(),
        backends=tuple(backend or ["lexicon"]),  # type: ignore[arg-type]
        label_source=label_source,  # type: ignore[arg-type]
    )
    console.print(f"[yellow]weak labels:[/yellow] {report['label_caveat']}")
    for name, metrics in report["backends"].items():
        _print_metrics(
            f"sentiment - {name}", {k: v for k, v in metrics.items() if not isinstance(v, dict)}
        )


@eval_app.command("signals")
def eval_signals_command(
    by: Annotated[str, typer.Option(help="Column defining the series.")] = "category",
    freq: Annotated[str, typer.Option(help="D | W")] = "D",
    sweep: Annotated[bool, typer.Option(help="Grid-search the thresholds.")] = True,
) -> None:
    """Run the weak-signal detector and score it against ground truth."""
    from itsm.pipelines.eval_weak_signal import run

    report = run(get_settings(), by=by, freq=freq, with_sweep=sweep)
    console.print(f"[bold]{report['n_alerts']}[/bold] alerts on {report['n_series']} series")
    evaluation = report.get("evaluation")
    if not evaluation:
        console.print("[yellow]no ground truth for this dataset - alerts are unscored[/yellow]")
        return
    _print_metrics(
        "detection quality", {k: v for k, v in evaluation.items() if not isinstance(v, list)}
    )
    table = Table(title="injected anomalies", header_style="bold")
    for column in ("category", "kind", "start", "detected", "first alert", "delay (days)"):
        table.add_column(column)
    for item in evaluation["per_anomaly"]:
        table.add_row(
            item["category"],
            item["kind"],
            item["start"],
            "yes" if item["detected"] else "no",
            str(item["first_alert"]),
            str(item["detection_delay_days"]),
        )
    console.print(table)


@app.command("report")
def show_report(
    name: Annotated[str, typer.Argument(help="classifier | sentiment | weak_signal | ...")],
) -> None:
    """Print a previously written report."""
    from itsm.pipelines.reporting import read_report

    report = read_report(name, get_settings())
    if report is None:
        console.print(f"[red]no report named {name!r}[/red]")
        raise typer.Exit(code=1)
    console.print_json(json.dumps(report)[:20000])


@app.command("app")
def run_streamlit(
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to bind.")] = 8501,
    headless: Annotated[bool, typer.Option(help="Do not open a browser.")] = False,
) -> None:
    """Start the Streamlit application.

    Bound to the loopback interface by default: the app displays ticket
    content, which is personal data, and Streamlit's own default would expose
    it on every interface of the machine. Pass ``--host 0.0.0.0`` deliberately.
    """
    import subprocess
    import sys
    from pathlib import Path

    import itsm

    # Resolved as a path, never imported: the module body *is* the page and
    # must only ever be executed by Streamlit itself.
    script = Path(itsm.__file__).parent / "app" / "main.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        host,
        "--server.port",
        str(port),
    ]
    if headless:
        command += ["--server.headless", "true"]
    raise typer.Exit(code=subprocess.call(command))


@app.command("serve")
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes.")] = False,
) -> None:
    """Start the FastAPI application."""
    import uvicorn

    uvicorn.run("itsm.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":  # pragma: no cover
    app()


# ------------------------------------------------------------------- privacy
@privacy_app.command("anonymise")
def privacy_anonymise(
    backend: Annotated[str | None, typer.Option(help="rules | transformer | presidio")] = None,
    keep_original: Annotated[
        bool, typer.Option(help="Also store the raw text as <column>_raw.")
    ] = False,
) -> None:
    """Produce an anonymised copy of the dataset."""
    from itsm.pipelines.anonymise import run

    report = run(
        get_settings(),
        backend=backend,  # type: ignore[arg-type]
        keep_original=keep_original,
    )
    _print_metrics(
        f"anonymisation ({report['backend']})",
        {
            key: report[key]
            for key in (
                "n_tickets",
                "tickets_with_pii",
                "tickets_with_pii_share",
                "spans_replaced",
                "distinct_values_replaced",
            )
        },
    )
    table = Table(title="personal data replaced by its entity tag", header_style="bold")
    table.add_column("entity")
    table.add_column("spans", justify="right")
    for entity, count in report["spans_by_entity"].items():
        table.add_row(entity, str(count))
    console.print(table)
    console.print(f"[green]written to[/green] {report['output']}")
    console.print("[yellow]Set ITSM_USE_ANONYMISED=true so every pipeline reads it.[/yellow]")


@privacy_app.command("eval")
def privacy_eval(
    backend: Annotated[
        list[str] | None, typer.Option(help="Repeatable: rules, transformer, presidio.")
    ] = None,
    column: Annotated[str, typer.Option(help="Free-text column to score.")] = "description",
    sample: Annotated[int, typer.Option(help="Tickets to score.")] = 2000,
    utility: Annotated[
        bool, typer.Option(help="Retrain the classifier to measure the cost.")
    ] = True,
    isolate: Annotated[
        bool,
        typer.Option(
            help="Score each backend on its own recognisers, with no rule layer underneath."
        ),
    ] = False,
) -> None:
    """Score anonymisation quality: detection, leakage and downstream cost."""
    from itsm.pipelines.eval_privacy import run

    report = run(
        get_settings(),
        backends=tuple(backend or ["rules"]),  # type: ignore[arg-type]
        column=column,
        sample_size=sample,
        with_utility=utility,
        isolate=isolate,
    )
    if not report["has_ground_truth"]:
        console.print(f"[yellow]{report['note']}[/yellow]")

    for name, entry in report["backends"].items():
        detection = entry.get("detection")
        if detection:
            _print_metrics(
                f"detection - {name}",
                {
                    key: detection[key]
                    for key in (
                        "recall_coverage",
                        "recall_partial",
                        "recall_strict",
                        "precision_partial",
                        "f1_partial",
                        "leak_rate_direct_identifiers",
                        "false_positive_spans",
                        "over_redacted_char_share",
                    )
                },
            )
            entities = Table(title=f"per entity - {name}", header_style="bold")
            for header in ("entity", "support", "recall", "precision"):
                entities.add_column(header)
            for entity, values in detection["per_entity"].items():
                entities.add_row(
                    entity,
                    str(values["support"]),
                    f"{values['recall']:.3f}",
                    f"{values['precision']:.3f}",
                )
            console.print(entities)

        if entry.get("residual"):
            _print_metrics(
                f"residual PII in the output - {name}",
                {
                    key: entry["residual"][key]
                    for key in (
                        "checked_values",
                        "residual_values",
                        "residual_rate",
                        "residual_rate_direct_identifiers",
                    )
                },
            )
        if entry.get("utility"):
            _print_metrics(f"downstream utility (macro-F1) - {name}", entry["utility"]["macro_f1"])

    probe = report.get("unseen_name_probe", {})
    if probe:
        table = Table(
            title=f"unseen-name probe ({probe['n_names']} names no backend has seen)",
            header_style="bold",
        )
        for header in ("backend", "recall", "missed probes"):
            table.add_column(header)
        for name, values in probe["backends"].items():
            table.add_row(name, f"{values['recall']:.3f}", str(values["missed_probes"]))
        console.print(table)
