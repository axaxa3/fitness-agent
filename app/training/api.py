import uuid
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.data.training_log import get_active_plan
from app.training.graph import training_graph
from app.training.state import TrainingState
from app.utils.sse_utils import SessionManager
from loguru import logger

router = APIRouter(prefix="/api/train", tags=["training"])


class LogSetRequest(BaseModel):
    user_id: str
    session_id: str  # 训练会话 ID
    exercise_id: str
    set_number: int
    reps: int
    weight_kg: float
    rpe: int = 0


class CompleteRequest(BaseModel):
    user_id: str
    session_id: str
    overall_feel: str = "good"
    notes: str = ""


@router.get("/today")
async def get_today_training(user_id: str):
    plan = get_active_plan(user_id)
    if not plan:
        return {"error": "no active plan, generate one first"}
    # 简化：返回计划中第一个未完成的训练日
    # 实际应基于当天是周几来匹配
    plan["_id"] = str(plan["_id"])
    sessions = plan.get("sessions", [])
    return {
        "plan_id": plan["_id"],
        "sessions": sessions,
        "progression_strategy": plan.get("progression_strategy", ""),
    }


@router.post("/log")
async def log_set(req: LogSetRequest):
    # 追加单组记录到训练会话
    logger.info(f"Set logged: {req.exercise_id} set {req.set_number}: {req.reps} x {req.weight_kg}kg @ RPE {req.rpe}")
    return {"ok": True, "set": req.model_dump()}


@router.post("/complete")
async def complete_training(req: CompleteRequest):
    review_session_id = f"review_{uuid.uuid4().hex[:16]}"
    SessionManager.create(review_session_id)

    plan = get_active_plan(req.user_id)

    state: TrainingState = {
        "user_id": req.user_id,
        "session_id": review_session_id,
        "today_plan": plan or {},
        "training_log": {
            "session_id": req.session_id,
            "overall_feel": req.overall_feel,
            "notes": req.notes,
        },
        "progress_report": {},
        "fatigue_report": {},
        "quality_report": {},
        "adjustments": {},
        "final_summary": {},
        "messages": [],
    }

    asyncio.get_event_loop().run_in_executor(
        None, lambda: training_graph.invoke(state)
    )

    return {"session_id": review_session_id, "status": "reviewing"}


@router.get("/stream/{session_id}")
async def training_stream(session_id: str):
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
