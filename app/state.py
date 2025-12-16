# app/state.py
from typing import Any, Dict, List
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # 入力
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]

    # コンテキスト（kintone 現在レコード）
    kintone_current_record: Dict[str, Any]

    # エージェントが作る「提案」
    kintone_updates: List[Dict[str, Any]]  # 例: {"field_code": "事前審査結果", "value": "否決"}
    notify_message: str

    # 管理用
    status: str              # "ready_for_review" 固定（HITL前の提案であることを示す）
    needs_human_review: bool # 現時点では True を返す（必ず人の確認が必要）

    applied: bool  # kintone 更新が適用済みかどうか
