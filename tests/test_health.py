from fastapi.testclient import TestClient

from app.main import app


def test_health_and_pages():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/").status_code == 200
        assert client.get("/sources").status_code == 200