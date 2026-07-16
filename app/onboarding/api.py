import uuid
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.onboarding.graph import onboarding_graph
from app.onboarding.state import OnboardingState
from app.utils.sse_utils import SessionManager
import asyncio
import json

router = APIRouter(prefix="/api/onboard", tags=["onboarding"])


class StartRequest(BaseModel):
    user_id: str | None = None


class MessageRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


@router.post("/start")
async def start_onboarding(req: StartRequest):
    user_id = req.user_id or f"user_{uuid.uuid4().hex[:12]}"
    session_id = f"onboard_{uuid.uuid4().hex[:16]}"
    return {
        "user_id": user_id,
        "session_id": session_id,
        "message": "你好！我是你的健身教练 🏋️ 在给你制定训练计划之前，先了解下你的情况——你之前有过健身经验吗？",
    }


@router.post("/message")
async def onboard_message(req: MessageRequest):
    session_id = req.session_id
    SessionManager.create(session_id)

    state: OnboardingState = {
        "user_id": req.user_id,
        "session_id": session_id,
        "messages": [],
        "user_input": req.message,
        "collected_info": {},
        "is_complete": False,
        "assistant_response": "",
    }

    asyncio.get_event_loop().run_in_executor(
        None, lambda: onboarding_graph.invoke(state)
    )

    return {"session_id": session_id, "status": "processing"}


@router.get("/stream/{session_id}")
async def onboard_stream(session_id: str):
    q = SessionManager.get(session_id)

    async def generate():
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': 'session not found'}})}\n\n"
            return
        while True:
            try:
                event = await asyncio.get_event_loop().run_in_executor(
                    None, q.get, True, 1
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
