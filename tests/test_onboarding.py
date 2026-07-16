"""Integration tests for the onboarding module.

Covers:
- POST /api/onboard/start – session creation
- POST /api/onboard/message – message processing dispatch
"""

from fastapi.testclient import TestClient


def test_onboard_start(client: TestClient):
    """Starting an onboarding session returns user_id, session_id, and a welcome message."""
    resp = client.post("/api/onboard/start", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert "user_id" in data
    assert "session_id" in data
    assert "message" in data


def test_onboard_message(client: TestClient):
    """Sending a message to an active onboarding session returns processing status."""
    # Start a session first
    start = client.post("/api/onboard/start", json={}).json()

    # Send a message
    resp = client.post(
        "/api/onboard/message",
        json={
            "user_id": start["user_id"],
            "session_id": start["session_id"],
            "message": "我是新手，想增肌，家里只有哑铃",
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "processing"
