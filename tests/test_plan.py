"""Integration tests for plan generation module."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_plan_generate_requires_user_id():
    """POST /api/plan/generate with missing user should return error."""
    resp = client.post("/api/plan/generate", json={"user_id": "nonexistent_user"})
    # user not found — should return an error response
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data


def test_plan_current_no_user():
    """GET /api/plan/current without user_id should return error."""
    resp = client.get("/api/plan/current?user_id=nonexistent")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
