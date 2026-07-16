from typing import TypedDict


class OnboardingState(TypedDict):
    user_id: str
    session_id: str
    messages: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]
    user_input: str
    collected_info: dict  # 逐步收集的用户信息
    is_complete: bool
    assistant_response: str
