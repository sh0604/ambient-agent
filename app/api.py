# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from .graph_app import graph_app
from .state import AgentState
from langgraph.types import Command

app = FastAPI()

class PreliminaryResultInput(BaseModel):
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]


# app/api.py
@app.post("/agent/propose-updates")
def propose_updates_endpoint(payload: PreliminaryResultInput):
    init_state: AgentState = {
        "anken_id": payload.anken_id,
        "mortgage_preliminary_result": payload.mortgage_preliminary_result,
    }

    result_state = graph_app.invoke(
        init_state,
        config={"configurable": {"thread_id": payload.anken_id}},
    )

    return {
        "anken_id": result_state["anken_id"],
        "kintone_updates": result_state.get("kintone_updates", []),
        "notify_message": result_state.get("notify_message", ""),
        "status": result_state.get("status", "unknown"),
    }

class StartInput(BaseModel):
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]

@app.post("/agent/start")
def start(payload: StartInput):
    init_state = {
        "anken_id": payload.anken_id,
        "mortgage_preliminary_result": payload.mortgage_preliminary_result,
    }

    # thread_id を案件単位で発行（簡易：anken_id）
    result = graph_app.invoke(
        init_state,
        config={"configurable": {"thread_id": payload.anken_id}},
    )

    # interrupt で止まっている場合、result は payload を含む
    return {
        "thread_id": payload.anken_id,
        "status": "REVIEW_REQUIRED",
        "review": result,  # interrupt payload
    }

class ResumeInput(BaseModel):
    thread_id: str
    action: str  # "approve" | "modify"
    kintone_updates: list | None = None

@app.post("/agent/resume")
def resume(payload: ResumeInput):
    command = Command(
        resume={
            "action": payload.action,
            "kintone_updates": payload.kintone_updates,
        }
    )

    result = graph_app.invoke(
        command,
        config={"configurable": {"thread_id": payload.thread_id}},
    )

    return {
        "thread_id": payload.thread_id,
        "status": result.get("status"),
        "applied": result.get("applied", False),
        "kintone_updates": result.get("kintone_updates"),
    }

