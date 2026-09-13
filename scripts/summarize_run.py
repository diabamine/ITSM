"""Print a compact summary of the reports produced by a full pipeline run."""

import json
from pathlib import Path

REPORTS = Path("reports")


def load(name: str) -> dict:
    """Read one report."""
    return json.loads((REPORTS / f"{name}.json").read_text(encoding="utf-8"))


def main() -> None:
    """Print the headline figures of every pipeline."""
    life = load("lifecycle_analysis")["overview"]
    print(
        "LIFECYCLE",
        {
            k: life[k]
            for k in (
                "tickets",
                "tickets_per_day",
                "open_tickets",
                "mttr_median_hours",
                "mttr_p90_hours",
                "sla_breach_rate",
                "reopen_rate",
                "first_time_right_rate",
            )
        },
    )

    clf = load("classifier")
    m = clf["metrics"]
    print(
        "CLASSIFIER",
        {k: m[k] for k in ("macro_f1", "accuracy", "balanced_accuracy", "top3_accuracy", "n_test")},
    )
    print("  baselines", {r["model"]: r["macro_f1"] for r in clf["model_comparison"]})
    print("  worst", [(c, s["f1"]) for c, s in m["worst_classes"]])
    print(
        "  coverage",
        [
            (p["threshold"], p["coverage"], p["accuracy"])
            for p in m["coverage_curve"]
            if p["threshold"] in (0.0, 0.4, 0.6, 0.8, 0.9)
        ],
    )

    des = load("designation_analysis")
    print("CLUSTERING", des["metrics"], des["taxonomy_alignment"])

    sen = load("sentiment")["backends"]["lexicon"]
    print("SENTIMENT", {k: v for k, v in sen.items() if not isinstance(v, dict)})

    sig = load("weak_signal")
    ev = sig["evaluation"]
    print(
        "SIGNALS",
        {
            k: ev[k]
            for k in (
                "n_alerts",
                "recall",
                "precision",
                "mean_detection_delay_days",
                "false_alarms_per_100_series_days",
            )
        },
    )
    for a in ev["per_anomaly"]:
        print(
            "   ",
            a["category"],
            a["kind"],
            "detected" if a["detected"] else "MISSED",
            a["detection_delay_days"],
            a["first_alert_kind"],
        )


if __name__ == "__main__":
    main()
