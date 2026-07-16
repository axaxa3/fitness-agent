import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress


def adjust_next_session(progress: dict, fatigue: dict, quality: dict, session_id: str) -> dict:
    push_progress(session_id, "正在综合生成下次训练调整...")
    llm = create_llm(temperature=0.3)
    prompt = f"""综合以下分析，生成下次训练的调整建议。
进度: {json.dumps(progress, ensure_ascii=False)}
疲劳: {json.dumps(fatigue, ensure_ascii=False)}
质量: {json.dumps(quality, ensure_ascii=False)}
返回JSON: {{"changes": [{{"type": "weight/sets/exercise_swap/rest", "detail": {{}}, "reason": ""}}], "summary": ""}}
只返回JSON。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
