# app/state.py
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # 入力
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]

    # コンテキスト（ローン申込アプリの現在レコード）
    kintone_current_record: Dict[str, Any]

    # ★追加：アプリ設計情報（フィールド定義など）
    kintone_app_schema: Dict[str, Any]  # { "app_id": "...", "fields": { field_code: {type,label,...}, ... } }

    # エージェントが作る「提案」
    kintone_updates: List[Dict[str, Any]]  # 例: {"field_code": "事前審査結果", "value": "否決"}
    notify_message: str

    # ★追加：LangSmith run_id（propose_updates の LLM 実行に紐づける）
    ls_run_id: str

    # ★追加：提案時点の原案（差分算出に使う）
    proposed_kintone_updates: List[Dict[str, Any]]
    proposed_notify_message: str

    # ★追加：簡易バリデーション結果
    validation_ok: bool
    validation_errors: List[str]

    # 管理用
    status: str              # "ready_for_review" 等
    needs_human_review: bool

    applied: bool  # kintone 更新が適用済みかどうか
    human_comment: Optional[str]
