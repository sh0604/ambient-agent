# app/nodes.py
from __future__ import annotations

import json
import logging
import os
from typing import TypedDict, Optional, Union, Literal, Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import Client
from langgraph.types import interrupt

from .state import AgentState

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-5.2")
ls_client = Client()

DATASET_ID = os.getenv("LANGCHAIN_DATASET_ID")


class HumanInterruptConfig(TypedDict):
    allow_ignore: bool
    allow_respond: bool
    allow_edit: bool
    allow_accept: bool


class ActionRequest(TypedDict):
    action: str
    args: Dict[str, Any]


class HumanInterrupt(TypedDict):
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: Optional[str]


class HumanResponse(TypedDict):
    type: Literal["accept", "ignore", "response", "edit"]
    args: Union[None, str, ActionRequest, Dict[str, Any]]


class RunIdCaptureHandler(BaseCallbackHandler):
    def __init__(self):
        self.run_id = None

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self.run_id = run_id


def _coerce_updates(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            loaded = json.loads(s)
        except json.JSONDecodeError:
            return []
        if isinstance(loaded, dict) and "kintone_updates" in loaded:
            loaded = loaded["kintone_updates"]
        if isinstance(loaded, list):
            return [v for v in loaded if isinstance(v, dict)]
        return []
    if isinstance(value, dict):
        return [value]
    return []


def _normalize_updates(updates: Any) -> list[dict[str, Any]]:
    updates_list = _coerce_updates(updates)
    return sorted(
        [{"field_code": u.get("field_code"), "value": u.get("value")} for u in updates_list],
        key=lambda x: (x.get("field_code") or ""),
    )


def _diff_updates(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    b = {u["field_code"]: u.get("value") for u in _normalize_updates(before) if u.get("field_code")}
    a = {u["field_code"]: u.get("value") for u in _normalize_updates(after) if u.get("field_code")}
    changed = []
    for fc in sorted(set(b.keys()) | set(a.keys())):
        if b.get(fc) != a.get(fc):
            changed.append({"field_code": fc, "before": b.get(fc), "after": a.get(fc)})
    return {"changed_fields": changed}


def _extract_edited_payload(resp_args: Any) -> dict[str, Any]:
    if resp_args is None:
        return {}

    if isinstance(resp_args, str):
        try:
            loaded = json.loads(resp_args)
            if isinstance(loaded, list):
                return {"kintone_updates": loaded}
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {}

    if isinstance(resp_args, dict):
        if "args" in resp_args and isinstance(resp_args["args"], dict):
            return resp_args["args"]
        return resp_args

    return {}


def _json_dumps_safe(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def _maybe_add_example_to_dataset(state: AgentState, diff: dict[str, Any]) -> None:
    if not DATASET_ID:
        logger.warning("[dataset] LANGCHAIN_DATASET_ID is not set; skip")
        return

    changed = diff.get("changed_fields", [])
    if not changed:
        return

    after_review_payload = {
        "kintone_updates": state.get("kintone_updates", []),
        "notify_message": state.get("notify_message", ""),
        "diff": diff,
    }
    after_review_str = _json_dumps_safe(after_review_payload)
    state["dataset_after_review"] = after_review_str

    inputs = {
        "bank": str(state.get("bank", "unknown")),
        "prompt": str(state.get("prompt_text", "")),
        "propose": str(state.get("dataset_propose_payload", "")),
        # ✅ ocr_content は mortgage_preliminary_result の JSON 文字列
        "ocr_content": str(state.get("ocr_content", "")),
        "after_review": after_review_str,
        "fetch_kintone_record": str(state.get("fetch_kintone_record", "")),
    }

    outputs = {
        "changed_fields_count": len(changed),
        "label": "diff_exists",
    }

    metadata = {
        "ls_run_id": state.get("ls_run_id"),
        "anken_id": state.get("anken_id"),
        "decision": state.get("status"),
        "bank_name": state.get("bank"),
    }

    try:
        ls_client.create_example(
            dataset_id=DATASET_ID,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        logger.info("[dataset] added example dataset_id=%s changed=%d", DATASET_ID, len(changed))
    except Exception:
        logger.exception("[dataset] failed to create_example")


def load_kintone_mock(state: AgentState) -> AgentState:
    anken_id = state["anken_id"]

    mock_record: Dict[str, Any] = {
        "案件番号": anken_id,
        "ローンフェーズ": "事前審査結果待ち",
        "事前審査結果": None,
        "事前審査結果受領日": None,
    }

    state["kintone_current_record"] = mock_record
    return state


def propose_updates(state: AgentState) -> AgentState:
    result = state["mortgage_preliminary_result"]
    record = state["kintone_current_record"]

    # ✅ ocr_content は mortgage_preliminary_result の JSON 文字列
    state["ocr_content"] = _json_dumps_safe(result)

    anken_id = state.get("anken_id")
    bank_name = (result or {}).get("金融機関名")
    schema_version = "loan_app_schema_v0"

    tags = [
        "node:propose_updates",
        f"bank:{bank_name}" if bank_name else "bank:unknown",
    ]
    metadata = {
        "anken_id": anken_id,
        "bank_name": bank_name,
        "schema_version": schema_version,
    }

    system = (
        "あなたは住宅ローン案件のオペレーション担当です。"
        "kintone の案件情報を、事前審査結果にしたがって更新する『提案』を JSON で出力してください。"
        "この出力はあくまで人間がレビューする前提の下書きであり、"
        "あなた自身がkintoneを直接更新することはありません。"
        "JSON 以外の文字は出力しないでください。"
    )

    user = f"""
kintone現在レコード:
{json.dumps(record, ensure_ascii=False)}

事前審査結果:
{json.dumps(result, ensure_ascii=False)}

出力フォーマットの例:
{{
  "kintone_updates": [
    {{"field_code": "事前審査結果", "value": "否決"}},
    {{"field_code": "事前審査結果受領日", "value": "2025-05-09"}},
    {{"field_code": "ローンフェーズ", "value": "事前審査結果受領済"}}
  ],
  "notify_message": "案件 ANKEN-123 の事前審査結果が『否決』でした。"
}}
"""

    # --- Dataset 用素材を state に保存 ---
    state["bank"] = bank_name or "unknown"
    state["prompt_text"] = "### system\n" + system + "\n\n### user\n" + user
    state["fetch_kintone_record"] = _json_dumps_safe(record)

    runid_handler = RunIdCaptureHandler()

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        config={"tags": tags, "metadata": metadata, "callbacks": [runid_handler]},
    )

    if runid_handler.run_id:
        state["ls_run_id"] = str(runid_handler.run_id)

    parsed = json.loads(resp.content)

    state["kintone_updates"] = parsed["kintone_updates"]
    state["notify_message"] = parsed["notify_message"]

    state["proposed_kintone_updates"] = parsed["kintone_updates"]
    state["proposed_notify_message"] = parsed["notify_message"]

    proposed_payload = {
        "kintone_updates": parsed["kintone_updates"],
        "notify_message": parsed["notify_message"],
    }
    state["dataset_propose_payload"] = _json_dumps_safe(proposed_payload)

    state["status"] = "ready_for_review"
    state["needs_human_review"] = True

    logger.info("[propose_updates] captured ls_run_id=%s", state.get("ls_run_id"))
    return state


def review_updates(state: AgentState) -> AgentState:
    req: HumanInterrupt = {
        "action_request": {
            "action": "ReviewKintoneUpdates",
            "args": {
                "anken_id": state["anken_id"],
                "kintone_updates": state["kintone_updates"],
                "notify_message": state["notify_message"],
            },
        },
        "config": {
            "allow_ignore": True,
            "allow_respond": True,
            "allow_edit": True,
            "allow_accept": True,
        },
        "description": "更新案を確認し、Accept / Edit / Respond / Ignore を選択してください。",
    }

    resp: HumanResponse = interrupt(req)[0]

    if resp["type"] == "ignore":
        state["status"] = "ignored"
        return state

    if resp["type"] == "response":
        state["human_comment"] = resp["args"]  # str想定
        state["status"] = "commented"
        return state

    proposed_updates = state.get("proposed_kintone_updates", state.get("kintone_updates", []))
    decision = resp["type"]  # "edit" or "accept"

    if resp["type"] == "edit":
        edited_payload = _extract_edited_payload(resp["args"])
        raw_updates = edited_payload.get("kintone_updates")
        raw_notify = edited_payload.get("notify_message")

        if raw_updates is not None:
            state["kintone_updates"] = _coerce_updates(raw_updates)
        if raw_notify is not None:
            state["notify_message"] = str(raw_notify)

        state["status"] = "edited"

    elif resp["type"] == "accept":
        state["status"] = "approved"

    else:
        state["status"] = "unknown_decision"
        return state

    proposed_updates_list = _coerce_updates(proposed_updates)
    final_updates_list = _coerce_updates(state.get("kintone_updates", []))
    final_notify = state.get("notify_message", "")

    diff = _diff_updates(proposed_updates_list, final_updates_list)

    state["review_diff"] = diff
    state["review_final"] = {
        "kintone_updates": final_updates_list,
        "notify_message": final_notify,
    }

    run_id = state.get("ls_run_id")
    if run_id:
        try:
            ls_client.create_feedback(run_id, key="human_decision", value=decision)
            ls_client.create_feedback(
                run_id,
                key="final_kintone_updates",
                value={"kintone_updates": final_updates_list, "notify_message": final_notify},
            )
            ls_client.create_feedback(run_id, key="update_diff", value=diff)
            ls_client.create_feedback(run_id, key="has_diff", value=bool(diff.get("changed_fields")))
        except Exception:
            logger.exception("[review_updates] failed to write LangSmith feedback")

    # ✅ diff がある場合だけ Dataset 登録
    _maybe_add_example_to_dataset(state, diff)

    return state


def apply_updates(state: AgentState) -> AgentState:
    state["applied"] = True
    state["status"] = "applied"
    return state


def finalize_output(state: AgentState) -> AgentState:
    state["applied"] = bool(state.get("applied", False))
    return state
