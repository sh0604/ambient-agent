# app/graph_app.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState
from .nodes import (
    load_kintone_mock,
    propose_updates,
    review_updates,
    apply_updates,
    finalize_output,
)

def build_graph():
    builder = StateGraph(AgentState)

    builder.add_node("load_kintone", load_kintone_mock)
    builder.add_node("propose_updates", propose_updates)
    builder.add_node("review_updates", review_updates)
    builder.add_node("apply_updates", apply_updates)
    builder.add_node("finalize_output", finalize_output)

    builder.add_edge(START, "load_kintone")
    builder.add_edge("load_kintone", "propose_updates")
    builder.add_edge("propose_updates", "review_updates")
    builder.add_edge("review_updates", "apply_updates")
    builder.add_edge("apply_updates", "finalize_output")
    builder.add_edge("finalize_output", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

graph_app = build_graph()
