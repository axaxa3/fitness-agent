"""Integration tests for training module."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_training_today_no_user():
    """GET /api/train/today without valid user should return error."""
    resp = client.get("/api/train/today?user_id=nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_training_complete_no_user():
    """POST /api/train/complete without valid user should still create review session."""
    resp = client.post("/api/train/complete", json={
        "user_id": "test_user",
        "session_id": "test_session_001",
        "overall_feel": "good",
        "notes": "test"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
