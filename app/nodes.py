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

def _get_recent_edit_examples(dataset_id: str, k: int = 3) -> list[dict[str, Any]]:
    # 例：decision が outputs に入っている前提
    # 取得件数は多め→ローカルで絞る方が安全
    examples = list(ls_client.list_examples(dataset_id=dataset_id, limit=50))
    edit_only = []
    for ex in examples:
        outputs = ex.outputs or {}
        if outputs.get("decision") == "edit":
            edit_only.append(ex)

    # created_atで新しい順に
    edit_only.sort(key=lambda e: getattr(e, "created_at", 0) or 0, reverse=True)
    return edit_only[:k]


def _summarize_edit_patterns(examples: list[Any]) -> str:
    # LLMに投げる用に整形（diff / final_payloadを中心に）
    items = []
    for ex in examples:
        inputs = ex.inputs or {}
        outputs = ex.outputs or {}
        items.append({
            "doc_type": inputs.get("doc_type"),
            "diff": outputs.get("diff") or inputs.get("diff"),
            "final_payload": inputs.get("final_payload"),
        })

    system = (
        "あなたは業務ルールの分析者です。"
        "以下は直近の人手編集（AI提案→人が修正した結果）の記録です。"
        "今後AIが提案を作る際に守るべき『編集傾向ルールTop3』を日本語で簡潔に箇条書きで出してください。"
        "曖昧なら『よく編集されるポイント』として一般化してよい。"
    )
    user = json.dumps(items, ensure_ascii=False)

    resp = llm.invoke([{"role": "system", "content": system},
                       {"role": "user", "content": user}])
    return resp.content.strip()


def _get_edit_patterns_summary(state: AgentState) -> str:
    if not DATASET_ID:
        return ""
    try:
        examples = _get_recent_edit_examples(DATASET_ID, k=3)
        if not examples:
            return ""
        return _summarize_edit_patterns(examples)
    except Exception:
        logger.exception("[patterns] failed to build edit patterns summary")
        return ""

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

    # --- 必須入力を schema に合わせて作る（すべて文字列で統一） ---
    anken_id = str(state.get("anken_id", ""))
    doc_type = str(state.get("doc_type", "preliminary_result"))  # 未設定なら暫定固定でOK

    kintone_current_record_str = _json_dumps_safe(state.get("kintone_current_record", {}))
    mortgage_preliminary_result_str = _json_dumps_safe(state.get("mortgage_preliminary_result", {}))

    prompt_text = str(state.get("prompt_text", ""))

    proposed_payload_str = str(state.get("dataset_propose_payload", ""))  # 既にJSON文字列
    # final_payload は review_final（確定値）を入れるのが自然
    final_payload_str = _json_dumps_safe(state.get("review_final", {
        "kintone_updates": state.get("kintone_updates", []),
        "notify_message": state.get("notify_message", "")
    }))

    human_comment = str(state.get("human_comment", "") or "")

    inputs = {
        "anken_id": anken_id,
        "doc_type": doc_type,
        "kintone_current_record": kintone_current_record_str,
        "mortgage_preliminary_result": mortgage_preliminary_result_str,
        "prompt_text": prompt_text,
        "proposed_payload": proposed_payload_str,
        "final_payload": final_payload_str,
        "human_comment": human_comment,
    }

    outputs = {
        "decision": "edit" if state.get("status") == "edited" else "accept",
        "has_diff": bool((diff or {}).get("changed_fields")),
        "diff": _json_dumps_safe(diff or {}),
    }

    metadata = {
        "ls_run_id": state.get("ls_run_id"),
        "anken_id": state.get("anken_id"),
        "status": state.get("status"),
        "schema_version": "loan_app_schema_v0",
    }

    try:
        ls_client.create_example(
            dataset_id=DATASET_ID,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        logger.info("[dataset] added example dataset_id=%s", DATASET_ID)
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

    patterns = _get_edit_patterns_summary(state)

    system = (
        "あなたは住宅ローン案件のオペレーション担当です。"
        "kintone の案件情報を、事前審査結果にしたがって更新する『提案』を JSON で出力してください。"
        "この出力はあくまで人間がレビューする前提の下書きであり、"
        "あなた自身がkintoneを直接更新することはありません。"
        "JSON 以外の文字は出力しないでください。"
        + ("\n\n【直近の人手編集傾向（参考）】\n" + patterns if patterns else "")
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

    resp: HumanResponse = interrupt(req)
    logger.info("[review_updates] human resp=%s", resp)

    run_id = state.get("ls_run_id")

    # ★ 上書き混乱を避けるため、keyを分ける（human_decision → review.decision）
    def _fb(key: str, value: Any) -> None:
        if not run_id:
            return
        try:
            ls_client.create_feedback(run_id, key=key, value=value)
        except Exception:
            logger.exception("[review_updates] failed to write feedback key=%s", key)

    rtype = resp.get("type")

    match rtype:
        case "ignore":
            state["status"] = "ignored"
            _fb("review.decision", "ignore")
            return state

        case "response":
            state["human_comment"] = resp.get("args")
            state["status"] = "commented"
            _fb("review.decision", "response")
            _fb("review.comment", state.get("human_comment", ""))
            return state

        case "accept" | "edit":
            decision = rtype  # accept or edit

        case _:
            state["status"] = "unknown_decision"
            _fb("review.decision", str(rtype))
            _fb("review.error", "unknown_decision")
            return state

    # --- ここから accept/edit のみ ---
    proposed_updates = state.get("proposed_kintone_updates", state.get("kintone_updates", []))

    if decision == "edit":
        edited_payload = _extract_edited_payload(resp.get("args"))
        raw_updates = edited_payload.get("kintone_updates")
        raw_notify = edited_payload.get("notify_message")

        if raw_updates is not None:
            state["kintone_updates"] = _coerce_updates(raw_updates)
        if raw_notify is not None:
            state["notify_message"] = str(raw_notify)

        state["status"] = "edited"
    else:
        state["status"] = "approved"

    proposed_updates_list = _coerce_updates(proposed_updates)
    final_updates_list = _coerce_updates(state.get("kintone_updates", []))
    final_notify = state.get("notify_message", "")

    diff = _diff_updates(proposed_updates_list, final_updates_list)

    state["review_diff"] = diff
    state["review_final"] = {
        "kintone_updates": final_updates_list,
        "notify_message": final_notify,
    }

    # Feedback（accept/edit）
    _fb("review.decision", decision)
    _fb("review.final", {"kintone_updates": final_updates_list, "notify_message": final_notify})
    _fb("review.diff", diff)
    _fb("review.has_diff", bool(diff.get("changed_fields")))

    # Dataset（accept/edit のみ）
    _maybe_add_example_to_dataset(state, diff)

    return state

def apply_updates(state: AgentState) -> AgentState:
    state["applied"] = True
    state["status"] = "applied"
    return state


def finalize_output(state: AgentState) -> AgentState:
    state["applied"] = bool(state.get("applied", False))
    return state
