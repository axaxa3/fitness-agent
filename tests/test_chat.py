"""Integration tests for the chat module.

Covers:
- POST /api/chat/message – Q&A session creation
- GET  /api/exercise/search – semantic exercise search
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


def test_chat_message(client: TestClient):
    """Posting a chat message returns a session_id and processing status."""
    resp = client.post(
        "/api/chat/message",
        json={
            "user_id": "test_user",
            "message": "深蹲膝盖疼怎么办",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data


@patch("app.tools.exercise_search.search_exercises_semantic")
def test_exercise_search(mock_search, client: TestClient):
    """Searching for exercises returns a results list.

    The underlying semantic search (Milvus) is mocked so the test
    can run without a running Milvus instance.
    """
    mock_search.return_value = [
        {
            "exercise_id": "goblet-squat",
            "name": "Goblet Squat",
            "name_cn": "高脚杯深蹲",
            "text": "高脚杯深蹲 Goblet Squat",
            "primary_muscles": ["股四头肌", "臀大肌"],
            "equipment": "哑铃",
            "difficulty": 2,
            "score": 0.92,
        },
    ]

    resp = client.get("/api/exercise/search?q=不伤膝盖的练腿动作")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert len(data["results"]) == 1
    assert data["results"][0]["exercise_id"] == "goblet-squat"
