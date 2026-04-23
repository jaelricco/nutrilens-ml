import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from nutrilens_ml.serve.app import app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_infer_plate_dev_mode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Dev env + empty secret = auth short-circuits.
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("SERVE_SHARED_SECRET", raising=False)

    r = client.post(
        "/infer/plate",
        json={"image_url": "https://example.com/a.jpg"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["task"] == "plate"
    assert 0 <= body["overall_confidence"] <= 1
    assert body["items"][0]["label"] == "pasta"


def test_infer_pour_liquid_validation(client: TestClient) -> None:
    r = client.post(
        "/infer/pour",
        json={"video_url": "https://example.com/a.mp4", "liquid": "not-a-liquid"},
    )
    assert r.status_code == 422
