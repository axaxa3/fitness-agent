from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class TrainingState(TypedDict):
    user_id: str
    session_id: str
    today_plan: dict
    training_log: dict
    progress_report: dict
    fatigue_report: dict
    quality_report: dict
    adjustments: dict
    final_summary: dict
    messages: Annotated[list, add_messages]
