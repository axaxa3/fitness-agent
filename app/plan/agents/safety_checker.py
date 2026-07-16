import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress
from app.data.exercise_library import get_exercise_by_id


def safety_check(full_plan: dict, user_profile: dict, session_id: str) -> dict:
    push_progress(session_id, "安全审核员正在审核计划安全性...")

    injury_flags = user_profile.get("injury_flags", [])
    plan_json = json.dumps(full_plan, ensure_ascii=False, indent=2)

    prompt = f"""你是训练安全审核员。审核以下训练计划，确保安全。

用户伤病标记：
{json.dumps(injury_flags, ensure_ascii=False)}

训练计划：
{plan_json}

检查项：
1. 是否有伤病标记中限制的动作？
2. 新手动作难度是否过高？
3. 同一肌群连续训练日是否有足够恢复时间（至少48小时）？
4. 总训练量是否合理（每个肌群每周10-20组为宜）？

返回JSON：
{{
  "passed": true/false,
  "issues": [
    {{"severity": "warning/error", "detail": "问题描述", "suggestion": "建议"}}
  ],
  "adjusted_plan": {{}} // 如果有修改，返回修改后的计划片段；如果passed=true，返回空对象
}}

只返回JSON，不要其他内容。"""

    llm = create_llm(temperature=0.1)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
