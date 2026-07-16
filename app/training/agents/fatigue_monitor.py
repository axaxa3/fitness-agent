import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress


def monitor_fatigue(training_log: dict, user_profile: dict, session_id: str) -> dict:
    push_progress(session_id, "疲劳监测员正在评估疲劳状态...")
    llm = create_llm(temperature=0.3)
    prompt = f"""你是疲劳监测员。评估用户疲劳状态。
训练日志: {json.dumps(training_log, ensure_ascii=False)}
用户历史RPE趋势: {json.dumps(user_profile.get('recent_rpe_avg', 0))}
返回JSON: {{"fatigue_level": "low/medium/high", "need_deload": true/false, "reason": "", "suggestion": ""}}
只返回JSON。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
