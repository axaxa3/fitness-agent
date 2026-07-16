import json
import time
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_delta


def synthesize_review(progress: dict, fatigue: dict, quality: dict, adjustments: dict, session_id: str) -> dict:
    llm = create_llm(temperature=0.3)
    prompt = f"""你是主教练。汇总复盘结果，给出最终训练反馈和建议。
进度: {json.dumps(progress, ensure_ascii=False)}
疲劳: {json.dumps(fatigue, ensure_ascii=False)}
质量: {json.dumps(quality, ensure_ascii=False)}
调整: {json.dumps(adjustments, ensure_ascii=False)}
返回JSON: {{"feedback": "对用户的鼓励和总结", "next_session_changes": "", "general_tips": ""}}
只返回JSON。"""
    response = llm.invoke([{"role": "user", "content": prompt}])
    result = json.loads(response.content)
    content_str = json.dumps(result, ensure_ascii=False)
    for i in range(0, len(content_str), 50):
        push_delta(session_id, content_str[i:i+50])
        time.sleep(0.02)
    return result
