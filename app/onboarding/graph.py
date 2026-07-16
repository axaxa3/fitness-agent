from langgraph.graph import StateGraph, END
from app.onboarding.state import OnboardingState
from app.core.llm_factory import create_llm
from app.core.prompt_loader import load_prompt_or_default
from app.data.user_profile import create_profile, mark_onboarding_complete
from app.utils.sse_utils import push_delta, push_final, push_error
from loguru import logger
import json

ONBOARDING_PROMPT = load_prompt_or_default(
    "onboarding",
    """你是健身教练，和新学员聊天收集信息。
需要收集: 健身经验、训练目标、可用器械、每周训练天数/时长、伤病情况。
{collected_info}
信息收集完后回答末尾加 [ONBOARDING_COMPLETE]""",
)

llm = create_llm(temperature=0.7)


def build_onboarding_graph() -> StateGraph:
    graph = StateGraph(OnboardingState)

    graph.add_node("chat", chat_node)
    graph.add_node("finalize", finalize_node)
    graph.set_entry_point("chat")
    graph.add_conditional_edges(
        "chat",
        route_after_chat,
        {"chat": "chat", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def chat_node(state: OnboardingState) -> dict:
    collected_json = json.dumps(state.get("collected_info", {}), ensure_ascii=False)
    prompt = ONBOARDING_PROMPT.replace("{collected_info}", collected_json)

    messages = [{"role": "system", "content": prompt}]
    for m in state.get("messages", []):
        messages.append(m)
    messages.append({"role": "user", "content": state["user_input"]})

    response = llm.invoke(messages)
    content = response.content

    assistant_msg = {"role": "assistant", "content": content}
    new_messages = list(state.get("messages", []))
    new_messages.append({"role": "user", "content": state["user_input"]})
    new_messages.append(assistant_msg)

    is_complete = "[ONBOARDING_COMPLETE]" in content
    if is_complete:
        content = content.replace("[ONBOARDING_COMPLETE]", "").strip()

    push_delta(state["session_id"], content)

    return {
        "messages": new_messages,
        "assistant_response": content,
        "is_complete": is_complete,
    }


def finalize_node(state: OnboardingState) -> dict:
    try:
        llm_local = create_llm(temperature=0.3)
        msgs = state.get("messages", [])
        extract_prompt = f"""根据以下对话，提取用户健身信息为JSON格式。
只返回JSON，不要其他内容：
{{"basic": {{"age": null, "gender": null, "height_cm": null, "weight_kg": null}},
 "fitness_profile": {{"goal": "", "level": "", "equipment": [], "days_per_week": 0, "session_minutes": 60}},
 "injury_notes": ""}}

对话记录：
{json.dumps(msgs[-6:], ensure_ascii=False)}"""

        result = llm_local.invoke([{"role": "user", "content": extract_prompt}])
        profile_data = json.loads(result.content)

        create_profile(state["user_id"], profile_data)
        mark_onboarding_complete(state["user_id"])

        push_final(state["session_id"], {
            "status": "complete",
            "message": "用户画像已生成，即将生成训练计划...",
            "profile": profile_data,
        })
        logger.info(f"Onboarding complete for user {state['user_id']}")
    except Exception as e:
        logger.error(f"Finalize error: {e}")
        push_error(state["session_id"], str(e))

    return {"assistant_response": "计划生成中..."}


def route_after_chat(state: OnboardingState) -> str:
    return "finalize" if state.get("is_complete") else "chat"


onboarding_graph = build_onboarding_graph()
