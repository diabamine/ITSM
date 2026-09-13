"""API contract tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from itsm.api.deps import reset_caches
from itsm.api.main import create_app


@pytest.fixture(scope="module")
def client(trained_workspace: Path) -> Iterator[TestClient]:
    """A test client backed by the trained temporary workspace."""
    reset_caches()
    with TestClient(create_app()) as test_client:
        yield test_client
    reset_caches()


def test_health_and_readiness(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["status"] == "ok"
    assert ready["artifacts"] == {"classifier": True, "clusterer": True}


def test_models_expose_their_metrics(client: TestClient) -> None:
    models = client.get("/v1/models").json()
    names = {model["name"] for model in models}
    assert {"classifier", "clusterer"} <= names
    classifier = next(model for model in models if model["name"] == "classifier")
    assert "macro_f1" in classifier["metrics"]


def test_classify_returns_labels_and_abstention(client: TestClient) -> None:
    response = client.post(
        "/v1/classify",
        json={"texts": ["Le VPN se deconnecte sans arret", "zzzz"], "min_confidence": 0.6},
    )
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["label"]
    assert len(payload["predictions"][0]["alternatives"]) == 3
    assert isinstance(payload["predictions"][1]["abstained"], bool)


def test_sentiment_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/sentiment", json={"texts": ["Delai inacceptable, je suis totalement bloque"]}
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["label"] == "negative"


def test_cluster_endpoints(client: TestClient) -> None:
    families = client.get("/v1/clusters").json()
    assert len(families) == 10
    assert all(family["top_terms"] for family in families)

    assigned = client.post("/v1/clusters/assign", json={"texts": ["Imprimante hors ligne"]}).json()
    assert assigned["results"][0]["cluster_id"] >= 0


def test_statistics_endpoints(client: TestClient) -> None:
    overview = client.get("/v1/stats/overview").json()
    assert overview["tickets"] > 0

    by_category = client.get("/v1/stats/by/category").json()
    assert len(by_category) > 1
    assert client.get("/v1/stats/by/not_a_column").status_code == 404

    flow = client.get("/v1/stats/flow?freq=W").json()
    assert flow["backlog"] and flow["status_updates"]["transitions"]


def test_ticket_trace(client: TestClient) -> None:
    assert client.get("/v1/tickets/INC0000001/trace").status_code == 200
    assert client.get("/v1/tickets/NOPE/trace").status_code == 404


def test_signal_scan_and_series(client: TestClient) -> None:
    scan = client.post("/v1/signals/scan", json={"by": "category", "freq": "D"})
    assert scan.status_code == 200
    payload = scan.json()
    assert payload["n_series"] > 1
    if payload["alerts"]:
        severities = [alert["severity"] for alert in payload["alerts"]]
        assert severities == sorted(severities, reverse=True)

    series = client.get("/v1/signals/series", params={"series": "Network & VPN"}).json()
    assert {"observed", "expected", "robust_z", "ewma"} <= set(series[0])
    assert client.get("/v1/signals/series", params={"series": "nope"}).status_code == 404


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/healthz", headers={"x-request-id": "abc-123"})
    assert response.headers["x-request-id"] == "abc-123"
    assert "x-process-time-ms" in response.headers


def test_validation_error_on_empty_batch(client: TestClient) -> None:
    assert client.post("/v1/classify", json={"texts": []}).status_code == 422
