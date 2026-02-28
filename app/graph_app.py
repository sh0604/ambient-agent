# app/graph_app.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # まずはこれでOK（PoC向け）

from .state import AgentState
from .nodes import load_kintone_mock, propose_updates, review_updates, apply_updates, finalize_output

def route_after_review(state: AgentState) -> str:
    status = state.get("status")
    if status in ("approved", "edited"):
        return "apply_updates"
    return "finalize_output"

def _build_builder():
    builder = StateGraph(AgentState)

    builder.add_node("load_kintone", load_kintone_mock)
    builder.add_node("propose_updates", propose_updates)
    builder.add_node("review_updates", review_updates)
    builder.add_node("apply_updates", apply_updates)
    builder.add_node("finalize_output", finalize_output)

    builder.add_edge(START, "load_kintone")
    builder.add_edge("load_kintone", "propose_updates")
    builder.add_edge("propose_updates", "review_updates")

    builder.add_conditional_edges(
        "review_updates",
        route_after_review,
        {"apply_updates": "apply_updates", "finalize_output": "finalize_output"},
    )
    builder.add_edge("apply_updates", "finalize_output")
    builder.add_edge("finalize_output", END)
    return builder

def build_graph_for_api():
    checkpointer = MemorySaver()
    return _build_builder().compile(checkpointer=checkpointer)

def build_graph_for_studio():
    # Studioで checkpointer が不要なら無しでもOK
    # もし Studio でも問題なければ、ここも同じ checkpointer を付けてよい
    return _build_builder().compile()

graph_api = build_graph_for_api()
graph_studio = build_graph_for_studio()

# Graph Studio 用のビルド関数をエクスポート
def build_graph():
    return graph_studio

# API 用の graph_app をエクスポート
graph_app = graph_api