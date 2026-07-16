import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress

llm = create_llm(temperature=0.3)


def design_split(user_profile: dict, session_id: str) -> dict:
    push_progress(session_id, "分化设计师正在分析最佳训练分化方式...")

    prompt = f"""你是训练分化设计师。根据用户信息，决定最佳训练分化方式。

用户信息：
{json.dumps(user_profile, ensure_ascii=False, indent=2)}

请选择并设计分化方案。可选：
- full_body: 全身训练（适合新手，每周2-3天）
- upper_lower: 上下肢分化（适合初中级，每周4天）
- ppl: 推拉腿分化（适合中级，每周3-6天）
- bro_split: 经典五分化（适合中高级有经验者，每周5天）

返回JSON：
{{
  "split_type": "upper_lower",
  "reason": "选择原因",
  "weekly_schedule": {{
    "day_1": {{"name": "上肢推", "focus": ["chest", "shoulders", "triceps"]}},
    "day_2": {{"name": "下肢", "focus": ["quads", "hamstrings", "glutes"]}},
    "day_3": {{"name": "上肢拉", "focus": ["back", "biceps", "rear_delts"]}},
    "day_4": {{"name": "下肢+核心", "focus": ["quads", "glutes", "core"]}}
  }},
  "sessions_per_week": 4
}}

只返回JSON，不要其他内容。"""

    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
