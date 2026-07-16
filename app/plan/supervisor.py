import json
from app.core.llm_factory import create_llm
from app.utils.sse_utils import push_progress, push_delta


def synthesize_plan(
    split: dict,
    exercises: dict,
    volume: dict,
    safety: dict,
    user_profile: dict,
    session_id: str,
) -> dict:
    push_progress(session_id, "主教练正在汇总生成最终训练计划...")

    prompt = f"""你是主教练。汇总以下专家意见，生成最终训练计划。

用户信息: {json.dumps(user_profile, ensure_ascii=False)}
分化方案: {json.dumps(split, ensure_ascii=False)}
动作选择: {json.dumps(exercises, ensure_ascii=False)}
负荷规划: {json.dumps(volume, ensure_ascii=False)}
安全审核: {json.dumps(safety, ensure_ascii=False)}

如果安全审核有修改建议，优先采纳安全审核的调整。

返回完整的训练计划JSON：
{{
  "split_type": "...",
  "cycle": "cycle_01",
  "week": 1,
  "sessions": [
    {{
      "day": "monday",
      "name": "...",
      "exercises": [
        {{
          "exercise_id": "...",
          "exercise_name": "...",
          "sets": 4,
          "target_reps": "8-10",
          "target_weight_kg": null,
          "rpe_target": 8,
          "rest_seconds": 90,
          "warmup_sets": 2,
          "order": 1,
          "notes": "最后一组做到力竭前1次"
        }}
      ]
    }}
  ],
  "progression_strategy": "...",
  "notes": "计划说明和注意事项",
  "decision_log": {{
    "split_reason": "...",
    "key_decisions": ["..."]
  }}
}}

只返回JSON，不要其他内容。"""

    llm = create_llm(temperature=0.3)
    response = llm.invoke([{"role": "user", "content": prompt}])

    import time
    plan_json = json.loads(response.content)
    content_str = json.dumps(plan_json, ensure_ascii=False, indent=2)

    # 流式推送最终计划
    chunk_size = 50
    for i in range(0, len(content_str), chunk_size):
        push_delta(session_id, content_str[i : i + chunk_size])
        time.sleep(0.02)

    return plan_json
