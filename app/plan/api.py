import uuid
import asyncio
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.plan.graph import plan_graph
from app.plan.state import PlanState
from app.data.user_profile import get_profile
from app.data.training_log import get_active_plan
from app.utils.sse_utils import SessionManager, push_error

router = APIRouter(prefix="/api/plan", tags=["plan"])


class GenerateRequest(BaseModel):
    user_id: str


class AdjustRequest(BaseModel):
    user_id: str
    request: str  # 用户用自然语言描述想要的调整


@router.post("/generate")
async def generate_plan(req: GenerateRequest):
    profile = get_profile(req.user_id)
    if not profile:
        return {"error": "user not found, complete onboarding first"}

    session_id = f"plan_{uuid.uuid4().hex[:16]}"
    SessionManager.create(session_id)

    state: PlanState = {
        "user_id": req.user_id,
        "session_id": session_id,
        "user_profile": profile,
        "split_suggestion": {},
        "exercise_selections": {},
        "volume_plan": {},
        "safety_report": {},
        "final_plan": {},
        "messages": [],
        "next_step": "design_split",
    }

    asyncio.get_event_loop().run_in_executor(
        None, lambda: plan_graph.invoke(state)
    )

    return {"session_id": session_id, "status": "processing"}


@router.get("/stream/{session_id}")
async def plan_stream(session_id: str):
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


@router.get("/current")
async def current_plan(user_id: str):
    plan = get_active_plan(user_id)
    if not plan:
        return {"error": "no active plan"}
    plan["_id"] = str(plan["_id"])
    return plan


@router.post("/adjust")
async def adjust_plan(req: AdjustRequest):
    # 复用 generate + 用户请求作为 extra 输入
    # MVP: 简单地重新生成计划，附带用户调整请求
    session_id = f"plan_adjust_{uuid.uuid4().hex[:16]}"
    SessionManager.create(session_id)

    profile = get_profile(req.user_id)
    if not profile:
        return {"error": "user not found"}

    # 将调整请求注入画像
    profile["_adjust_request"] = req.request

    state: PlanState = {
        "user_id": req.user_id,
        "session_id": session_id,
        "user_profile": profile,
        "split_suggestion": {},
        "exercise_selections": {},
        "volume_plan": {},
        "safety_report": {},
        "final_plan": {},
        "messages": [],
        "next_step": "design_split",
    }

    asyncio.get_event_loop().run_in_executor(
        None, lambda: plan_graph.invoke(state)
    )

    return {"session_id": session_id, "status": "processing"}
