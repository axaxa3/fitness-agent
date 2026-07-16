from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class QAState(TypedDict):
    user_id: str
    session_id: str
    question: str
    retrieved_docs: list[dict]
    context: str
    answer: str
    messages: Annotated[list, add_messages]
