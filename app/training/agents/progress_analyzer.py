import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress


def analyze_progress(training_log: dict, session_id: str) -> dict:
    push_progress(session_id, "进度分析师正在评估训练完成情况和进步趋势...")
    llm = create_llm(temperature=0.3)
    prompt = f"""你是训练进度分析师。分析以下训练日志：
{json.dumps(training_log, ensure_ascii=False)}
返回JSON: {{"volume_muscles": {{}}, "prs_hit": [], "completion_rate": 0.0, "summary": ""}}
只返回JSON。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
