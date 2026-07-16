import uuid
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.chat.graph import chat_graph
from app.chat.state import QAState
from app.tools.exercise_search import search_exercises_semantic
from app.utils.sse_utils import SessionManager

router = APIRouter(prefix="/api/chat", tags=["chat"])


class ChatRequest(BaseModel):
    user_id: str = "anonymous"
    message: str


@router.post("/message")
async def chat_message(req: ChatRequest):
    session_id = f"chat_{uuid.uuid4().hex[:16]}"
    SessionManager.create(session_id)

    state: QAState = {
        "user_id": req.user_id,
        "session_id": session_id,
        "question": req.message,
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "messages": [],
    }

    asyncio.get_event_loop().run_in_executor(
        None, lambda: chat_graph.invoke(state)
    )

    return {"session_id": session_id, "status": "processing"}


@router.get("/stream/{session_id}")
async def chat_stream(session_id: str):
    q = SessionManager.get(session_id)

    async def generate():
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'session not found'}})}\n\n"
            return
        while True:
            try:
                event = await asyncio.get_event_loop().run_in_executor(
                    None, q.get, True, 5
                )
                yield f"event: {event['type']}\ndata: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                if event["type"] in ("final", "error"):
                    break
            except Exception:
                break
        SessionManager.remove(session_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/exercise/search")
async def search_exercises(q: str, equipment: str | None = None, difficulty: int | None = None):
    filters = {}
    if equipment:
        filters["equipment"] = equipment
    if difficulty:
        filters["difficulty"] = difficulty
    results = search_exercises_semantic(q, top_k=10, filters=filters or None)
    return {"results": results}
