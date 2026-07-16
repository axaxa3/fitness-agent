from langgraph.graph import StateGraph, END
from app.plan.state import PlanState
from app.plan.agents.split_designer import design_split
from app.plan.agents.exercise_selector import select_exercises
from app.plan.agents.volume_planner import plan_volume
from app.plan.agents.safety_checker import safety_check
from app.plan.supervisor import synthesize_plan
from app.data.training_log import create_plan, deactivate_plan
from app.utils.sse_utils import push_final, push_error
from loguru import logger


def build_plan_graph() -> StateGraph:
    graph = StateGraph(PlanState)

    graph.add_node("design_split", design_split_node)
    graph.add_node("select_exercises", select_exercises_node)
    graph.add_node("plan_volume", plan_volume_node)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("design_split")
    graph.add_edge("design_split", "select_exercises")
    graph.add_edge("select_exercises", "plan_volume")
    graph.add_edge("plan_volume", "safety_check")
    graph.add_edge("safety_check", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def design_split_node(state: PlanState) -> dict:
    split = design_split(state["user_profile"], state["session_id"])
    return {"split_suggestion": split}


def select_exercises_node(state: PlanState) -> dict:
    exercises = select_exercises(
        state["split_suggestion"], state["user_profile"], state["session_id"]
    )
    return {"exercise_selections": exercises}


def plan_volume_node(state: PlanState) -> dict:
    volume = plan_volume(
        state["exercise_selections"],
        state["user_profile"],
        state["split_suggestion"],
        state["session_id"],
    )
    return {"volume_plan": volume}


def safety_check_node(state: PlanState) -> dict:
    safety = safety_check(
        {
            "split": state["split_suggestion"],
            "exercises": state["exercise_selections"],
            "volume": state["volume_plan"],
        },
        state["user_profile"],
        state["session_id"],
    )
    return {"safety_report": safety}


def synthesize_node(state: PlanState) -> dict:
    try:
        plan = synthesize_plan(
            state["split_suggestion"],
            state["exercise_selections"],
            state["volume_plan"],
            state["safety_report"],
            state["user_profile"],
            state["session_id"],
        )

        # 停用旧计划，存入新计划
        user_id = state["user_id"]
        old_plan = None
        try:
            from app.data.training_log import get_active_plan
            old_plan = get_active_plan(user_id)
            if old_plan:
                deactivate_plan(old_plan["_id"])
        except Exception:
            pass

        create_plan(user_id, plan)

        push_final(state["session_id"], {
            "status": "complete",
            "plan": plan,
        })
        logger.info(f"Plan generated for user {user_id}")

        return {"final_plan": plan}

    except Exception as e:
        logger.error(f"Plan synthesis error: {e}")
        push_error(state["session_id"], str(e))
        return {"final_plan": {}}


plan_graph = build_plan_graph()
