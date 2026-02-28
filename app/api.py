from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Literal, Union
from uuid import uuid4

from .graph_app import graph_app
from .state import AgentState
from langgraph.types import Command

app = FastAPI()

class StartInput(BaseModel):
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]

def _is_interrupt_payload(obj: Any) -> bool:
    # review_updates が interrupt(req) に渡している dict 形式を想定
    return isinstance(obj, dict) and "action_request" in obj and "config" in obj

def _is_interrupt_payload(x: Any) -> bool:
    return isinstance(x, dict) and "action_request" in x and "config" in x

@app.post("/agent/start")
def start(payload: StartInput):
    thread_id = str(uuid4())

    result = graph_app.invoke(
        {"anken_id": payload.anken_id, "mortgage_preliminary_result": payload.mortgage_preliminary_result},
        config={"configurable": {"thread_id": thread_id}},
    )

    if _is_interrupt_payload(result):
        return {"thread_id": thread_id, "status": "REVIEW_REQUIRED", "review": result}

    if isinstance(result, dict):
        return {"thread_id": thread_id, "status": result.get("status"), "result": result}

    raise HTTPException(500, "Unexpected graph result type")

class ResumeInput(BaseModel):
    thread_id: str
    type: Literal["accept", "edit", "response", "ignore"]
    args: Optional[Union[None, str, Dict[str, Any]]] = None
    # edit のときは args に {"kintone_updates": [...], "notify_message": "..."} を入れる想定


@app.post("/agent/resume")
def resume(payload: ResumeInput):
    human_response = {"type": payload.type, "args": payload.args}

    result = graph_app.invoke(
        Command(resume=human_response),
        config={"configurable": {"thread_id": payload.thread_id}},
    )

    # resume 後は基本 state が返る想定
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Unexpected graph result type after resume")

    return {
        "thread_id": payload.thread_id,
        "status": result.get("status"),
        "applied": bool(result.get("applied", False)),
        "anken_id": result.get("anken_id"),
        "kintone_updates": result.get("kintone_updates", []),
        "notify_message": result.get("notify_message", ""),
        "review_diff": result.get("review_diff"),
    }
