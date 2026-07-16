from langgraph.graph import StateGraph, END
from app.training.state import TrainingState
from app.training.agents.progress_analyzer import analyze_progress
from app.training.agents.fatigue_monitor import monitor_fatigue
from app.training.agents.quality_assessor import assess_quality
from app.training.agents.next_session_adjuster import adjust_next_session
from app.training.supervisor import synthesize_review
from app.utils.sse_utils import push_final, push_error
from loguru import logger


def build_training_graph() -> StateGraph:
    graph = StateGraph(TrainingState)
    graph.add_node("analyze_progress", progress_node)
    graph.add_node("monitor_fatigue", fatigue_node)
    graph.add_node("assess_quality", quality_node)
    graph.add_node("adjust_next", adjust_node)
    graph.add_node("synthesize_review", review_node)
    graph.set_entry_point("analyze_progress")
    graph.add_edge("analyze_progress", "monitor_fatigue")
    graph.add_edge("monitor_fatigue", "assess_quality")
    graph.add_edge("assess_quality", "adjust_next")
    graph.add_edge("adjust_next", "synthesize_review")
    graph.add_edge("synthesize_review", END)
    return graph.compile()


def progress_node(state: TrainingState) -> dict:
    return {"progress_report": analyze_progress(state["training_log"], state["session_id"])}

def fatigue_node(state: TrainingState) -> dict:
    return {"fatigue_report": monitor_fatigue(state["training_log"], {}, state["session_id"])}

def quality_node(state: TrainingState) -> dict:
    return {"quality_report": assess_quality(state["training_log"], state["session_id"])}

def adjust_node(state: TrainingState) -> dict:
    return {"adjustments": adjust_next_session(
        state["progress_report"], state["fatigue_report"], state["quality_report"], state["session_id"]
    )}

def review_node(state: TrainingState) -> dict:
    try:
        summary = synthesize_review(
            state["progress_report"], state["fatigue_report"],
            state["quality_report"], state["adjustments"], state["session_id"]
        )
        push_final(state["session_id"], {"status": "complete", "summary": summary})
    except Exception as e:
        logger.error(f"Review error: {e}")
        push_error(state["session_id"], str(e))
    return {"final_summary": summary}


training_graph = build_training_graph()
