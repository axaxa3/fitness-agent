from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class PlanState(TypedDict):
    user_id: str
    session_id: str
    user_profile: dict
    split_suggestion: dict
    exercise_selections: dict
    volume_plan: dict
    safety_report: dict
    final_plan: dict
    messages: Annotated[list, add_messages]
    next_step: str
