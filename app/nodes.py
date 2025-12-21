# app/nodes.py
from typing import TypedDict, Optional, Union, Literal, Dict, Any
from .state import AgentState
from langchain_openai import ChatOpenAI
import json
import logging

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-5.2")

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

    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )

    parsed = json.loads(resp.content)

    state["kintone_updates"] = parsed["kintone_updates"]
    state["notify_message"] = parsed["notify_message"]

    # ここで「HITL 前の提案である」ことを明示する
    state["status"] = "ready_for_review"
    state["needs_human_review"] = True

    logger.info(f"[propose_updates] kintone_updates proposal: {state['kintone_updates']}")
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

    if resp["type"] == "edit":
        # edit の場合、args は ActionRequest（action + args）になる想定
        # ここでは「修正後の kintone_updates」を受け取る設計にする
        edited = resp["args"]
        if isinstance(edited, dict) and "args" in edited and "kintone_updates" in edited["args"]:
            state["kintone_updates"] = edited["args"]["kintone_updates"]
        state["status"] = "edited"
        return state

    # accept
    state["status"] = "approved"
    return state

def apply_updates(state: AgentState) -> AgentState:
    # TODO: 次タスク②で実kintone APIに差し替え
    state["applied"] = True
    state["status"] = "applied"
    return state

