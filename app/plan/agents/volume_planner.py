import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress
from app.data.exercise_library import get_exercise_by_id


def plan_volume(
    exercise_selections: dict, user_profile: dict, split: dict, session_id: str
) -> dict:
    push_progress(session_id, "负荷规划师正在设定组数、次数、重量和渐进方案...")

    exercises_detail = {}
    for day_key, exercises in exercise_selections.items():
        for ex in exercises:
            detail = get_exercise_by_id(ex["exercise_id"])
            if detail:
                exercises_detail[ex["exercise_id"]] = {
                    "name_cn": detail["name_cn"],
                    "category": detail["category"],
                    "rep_range": detail.get("rep_range", [8, 12]),
                }

    prompt = f"""你是训练负荷规划师。为每个动作设定组数、次数、重量和渐进方案。

用户水平: {user_profile.get('fitness_profile', {}).get('level', 'beginner')}
用户目标: {user_profile.get('fitness_profile', {}).get('goal', 'muscle_gain')}
每次训练时长: {user_profile.get('fitness_profile', {}).get('session_minutes', 60)}分钟

训练日安排：
{json.dumps(exercise_selections, ensure_ascii=False, indent=2)}

动作详情：
{json.dumps(exercises_detail, ensure_ascii=False, indent=2)}

规则：
- 增肌(hypertrophy): 组数3-4组，次数8-12，RPE 7-9
- 减脂(fat_loss): 组数3-4组，次数12-20，RPE 6-8，组间休息45-60秒
- 力量(strength): 组数4-5组，次数3-6，RPE 8-9，组间休息2-3分钟
- 新手从较轻重量开始，每周增加2.5-5kg
- 每个动作至少2个热身组

为每个训练日返回：
{{
  "day_1": [
    {{
      "exercise_id": "...",
      "sets": 4,
      "target_reps": "8-10",
      "target_weight_kg": null,
      "rpe_target": 8,
      "rest_seconds": 90,
      "warmup_sets": 2,
      "progression": "每周增加2.5kg，当能完成目标次数上限时加重量"
    }},
    ...
  ],
  "progression_strategy": "线性渐进：每周每个动作增加2.5-5kg或额外1次。每4周安排一次减载周（容量减半）。"
}}

只返回JSON，不要其他内容。"""

    llm = create_llm(temperature=0.3)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return json.loads(response.content)
