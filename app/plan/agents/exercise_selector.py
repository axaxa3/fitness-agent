import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress
from app.data.exercise_library import list_all_exercises



def select_exercises(split: dict, user_profile: dict, session_id: str) -> dict:
    push_progress(session_id, "动作专家正在为每个训练日挑选最佳动作...")

    all_exercises = list_all_exercises()
    exercise_summary = "\n".join([
        f"- {ex['exercise_id']}: {ex['name_cn']} | 主肌群: {', '.join(ex['primary_muscles'])} | "
        f"器械: {ex['equipment']} | 难度: {ex['difficulty']} | 分类: {ex['category']} | "
        f"禁忌: {', '.join(ex.get('contraindications', []))}"
        for ex in all_exercises
    ])

    injury_flags = user_profile.get("injury_flags", [])
    restricted = []
    for f in injury_flags:
        restricted.extend(f.get("restricted_exercise_ids", []))

    prompt = f"""你是动作选择专家。为每个训练日选择最适合的动作组合。

用户可用器械：{user_profile.get('fitness_profile', {}).get('equipment', [])}
用户水平：{user_profile.get('fitness_profile', {}).get('level', 'beginner')}
限制动作（伤病）：{restricted}
训练频率：{split.get('sessions_per_week')}

分化方案：
{json.dumps(split, ensure_ascii=False, indent=2)}

可用动作库：
{exercise_summary}

规则：
1. 每个训练日选5-7个动作
2. 先排复合动作（compound），再排孤立动作（isolation）
3. 覆盖当天的所有目标肌群
4. 不选 restricted 列表中的动作
5. 新手（beginner）优先选难度1-2的动作，中级（intermediate）可混选，高级（advanced）可全选
6. 确保每个主要肌群每周至少被直接训练2次

为每个训练日返回：
{{
  "day_1": [
    {{"exercise_id": "...", "order": 1, "reason": "选择原因"}},
    ...
  ],
  ...
}}

只返回JSON，不要其他内容。"""

    llm = create_llm(temperature=0.3)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
