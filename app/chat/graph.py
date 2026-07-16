from langgraph.graph import StateGraph, END
from app.chat.state import QAState
from app.core.llm_factory import create_llm
from app.core.prompt_loader import load_prompt_or_default
from app.tools.exercise_search import search_exercises_semantic
from app.utils.sse_utils import push_delta, push_final, push_error
from loguru import logger
import time

CHAT_PROMPT = load_prompt_or_default(
    "chat",
    "你是健身教练。使用以下知识回答问题。\n知识：{context}\n问题：{question}\n诚实、专业、积极。",
)

llm = create_llm(temperature=0.7)


def build_chat_graph() -> StateGraph:
    graph = StateGraph(QAState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


def retrieve_node(state: QAState) -> dict:
    question = state["question"]
    docs = search_exercises_semantic(question, top_k=3)
    context = "\n\n".join([
        f"动作: {d['name_cn']}({d['name']})\n目标肌群: {', '.join(d.get('primary_muscles') or [])}\n{d.get('text', '')}"
        for d in docs
    ])
    return {"retrieved_docs": docs, "context": context or "无相关知识"}


def answer_node(state: QAState) -> dict:
    content = ""
    try:
        prompt = CHAT_PROMPT.format(
            context=state["context"],
            question=state["question"],
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        content = response.content

        for i in range(0, len(content), 20):
            push_delta(state["session_id"], content[i : i + 20])
            time.sleep(0.02)

        push_final(state["session_id"], {
            "status": "complete",
            "answer": content,
            "references": [
                {"name": d["name_cn"], "exercise_id": d["exercise_id"]}
                for d in state.get("retrieved_docs", [])[:3]
            ],
        })
    except Exception as e:
        logger.error(f"Chat error: {e}")
        push_error(state["session_id"], str(e))

    return {"answer": content}


chat_graph = build_chat_graph()
