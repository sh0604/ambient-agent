# app/nodes.py
from typing import TypedDict, Optional, Union, Literal, Dict, Any
from .state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langsmith import Client
import json
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-5.2")

ls_client = Client()

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
    """
    kintone_updates を list[dict] に正規化する。
    - すでに list[dict] → そのまま
    - JSON文字列 → json.loads して解釈
    - その他 → 空にする（PoCは安全側）
    """
    if value is None:
        return []
    if isinstance(value, list):
        # list[str] など混在の可能性があるので dict だけ残す
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            loaded = json.loads(s)
        except json.JSONDecodeError:
            return []
        # {"kintone_updates":[...]} の形で返る可能性も吸収
        if isinstance(loaded, dict) and "kintone_updates" in loaded:
            loaded = loaded["kintone_updates"]
        if isinstance(loaded, list):
            return [v for v in loaded if isinstance(v, dict)]
        return []
    # dict単体で来るケースも念のため
    if isinstance(value, dict):
        return [value]
    return []

def _normalize_updates(updates: Any) -> list[dict[str, Any]]:
    # field_code で安定ソートして差分を取りやすくする（PoC用）
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
    """
    Agent Inbox から返ってくる edit payload の揺れを吸収して、
    {"kintone_updates": ..., "notify_message": ...} の形に正規化する。
    """
    if resp_args is None:
        return {}

    # 1) 文字列で JSON が来るケース
    if isinstance(resp_args, str):
        try:
            loaded = json.loads(resp_args)
            # list なら kintone_updates とみなす
            if isinstance(loaded, list):
                return {"kintone_updates": loaded}
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            return {}

    # 2) dict のケース
    if isinstance(resp_args, dict):
        # ActionRequest 形式 {"action": "...", "args": {...}}
        if "args" in resp_args and isinstance(resp_args["args"], dict):
            return resp_args["args"]
        # すでに args 本体が入っている
        return resp_args

    return {}

def load_kintone_mock(state: AgentState) -> AgentState:
    """anken_id から案件情報をモック取得するノード。
    ここではまだ kintone 本体は更新しない。
    """
    anken_id = state["anken_id"]

    mock_record: Dict[str, Any] = {
        "案件番号": anken_id,
        "ローンフェーズ": "事前審査結果待ち",
        "事前審査結果": None,
        "事前審査結果受領日": None,
    }

    state["kintone_current_record"] = mock_record
    # この段階ではまだ案もレビューも未実施
    return state


def propose_updates(state: AgentState) -> AgentState:
    """事前審査結果 + 現在の案件情報 から
    kintone 更新案（提案）を作るノード。
    ここではあくまで「案」を作るだけで、kintone 更新は行わない。
    """
    result = state["mortgage_preliminary_result"]
    record = state["kintone_current_record"]

    # ★追加：trace用のメタデータ（PoCは固定でOK）
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
    
    runid_handler = RunIdCaptureHandler()

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        config={"tags": tags, "metadata": metadata, "callbacks": [runid_handler],},
    )

    if runid_handler.run_id:
        state["ls_run_id"] = str(runid_handler.run_id)

    parsed = json.loads(resp.content)

    state["kintone_updates"] = parsed["kintone_updates"]
    state["notify_message"] = parsed["notify_message"]

    state["proposed_kintone_updates"] = parsed["kintone_updates"]
    state["proposed_notify_message"] = parsed["notify_message"]

    # ここで「HITL 前の提案である」ことを明示する
    state["status"] = "ready_for_review"
    state["needs_human_review"] = True

    logger.info(f"[propose_updates] kintone_updates proposal: {state['kintone_updates']}")
    logger.info(f"[propose_updates] captured ls_run_id={state.get('ls_run_id')}")
    return state


def finalize_output(state: AgentState) -> AgentState:
    """APIレスポンスとして返しやすい形を整えるノード。
    今回は state をそのまま返すだけ。
    将来的にマスクや余計な情報の削除をここで行う。
    """
    state["applied"] = bool(state.get("applied", False))
    return state

# app/nodes.py
from langgraph.types import interrupt
from .state import AgentState

def review_updates(state: AgentState) -> AgentState:
    # Agent Inbox に「何をレビューしてほしいか」を action_request として渡す
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

    # interrupt() は HumanResponse の配列を返す想定（Inbox UI は長さ1が前提）
    resp: HumanResponse = interrupt(req)[0]

    if resp["type"] == "ignore":
        state["status"] = "ignored"
        return state

    if resp["type"] == "response":
        # 例：コメントだけ保存したい場合
        state["human_comment"] = resp["args"]  # str を想定
        state["status"] = "commented"
        return state
    
    # ★ここから：edit / accept を LangSmith feedback に記録する
    proposed_updates = state.get("proposed_kintone_updates", state.get("kintone_updates", []))
    proposed_notify = state.get("proposed_notify_message", state.get("notify_message", ""))

    decision = resp["type"]  # "edit" or "accept" になる想定

    if resp["type"] == "edit":
        edited_payload = _extract_edited_payload(resp["args"])
        raw_updates = edited_payload.get("kintone_updates")
        raw_notify = edited_payload.get("notify_message")

        logger.info(
            "[review_updates] raw kintone_updates type=%s snippet=%s",
            type(raw_updates).__name__,
            (str(raw_updates)[:200] if raw_updates is not None else "None"),
        )

        if raw_updates is not None:
            state["kintone_updates"] = _coerce_updates(raw_updates)
        if raw_notify is not None:
            state["notify_message"] = str(raw_notify)

        logger.info(
            "[review_updates] coerced kintone_updates=%s",
            state.get("kintone_updates"),
        )

        state["status"] = "edited"

    elif resp["type"] == "accept":
        state["status"] = "approved"

    else:
        state["status"] = "unknown_decision"
        return state
    
    final_updates = state.get("kintone_updates", [])
    final_notify = state.get("notify_message", "")
    
    logger.info(
        f"[review_updates] will_write_feedback run_id={state.get('ls_run_id')} decision={decision}"
    )

    run_id = state.get("ls_run_id")
    if run_id:
        try:
            proposed_updates = _coerce_updates(proposed_updates)
            final_updates = _coerce_updates(final_updates)

            diff = _diff_updates(proposed_updates, final_updates)

            ls_client.create_feedback(run_id, key="human_decision", value=decision)
            ls_client.create_feedback(
                run_id,
                key="final_kintone_updates",
                value={"kintone_updates": final_updates, "notify_message": final_notify},
            )
            ls_client.create_feedback(run_id, key="update_diff", value=diff)

        except Exception:
            logger.exception("[review_updates] failed to write LangSmith feedback")

    return state

def apply_updates(state: AgentState) -> AgentState:
    # TODO: 次タスク②で実kintone APIに差し替え
    state["applied"] = True
    state["status"] = "applied"
    return state

