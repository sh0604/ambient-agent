# app/state.py
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    # 入力
    anken_id: str
    mortgage_preliminary_result: Dict[str, Any]

    # ocr_content（= mortgage_preliminary_result を JSON 文字列化したもの）
    ocr_content: str

    # コンテキスト（ローン申込アプリの現在レコード）
    kintone_current_record: Dict[str, Any]

    # アプリ設計情報（フィールド定義など）
    kintone_app_schema: Dict[str, Any]

    # エージェントが作る「提案」
    kintone_updates: List[Dict[str, Any]]
    notify_message: str

    # LangSmith run_id
    ls_run_id: str

    # 提案時点の原案（差分算出に使う）
    proposed_kintone_updates: List[Dict[str, Any]]
    proposed_notify_message: str

    # レビュー結果
    review_diff: Dict[str, Any]
    review_final: Dict[str, Any]

    # --- Dataset登録用（あなたの input schema に対応）---
    bank: str
    prompt_text: str
    dataset_propose_payload: str
    fetch_kintone_record: str
    dataset_after_review: str

    # 管理用
    status: str
    needs_human_review: bool

    applied: bool
    human_comment: Optional[str]
