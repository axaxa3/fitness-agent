import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress


def assess_quality(training_log: dict, session_id: str) -> dict:
    push_progress(session_id, "动作质量评估员正在分析训练质量...")
    llm = create_llm(temperature=0.3)
    prompt = f"""你是动作质量评估员。分析训练质量和用户反馈。
训练日志: {json.dumps(training_log, ensure_ascii=False)}
关注: 是否有疼痛/不适反馈？动作完成度如何？
返回JSON: {{"issues": [{{"exercise_id": "", "problem": "", "action": "replace/reduce_weight/monitor"}}], "safe_to_continue": true}}
只返回JSON。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
