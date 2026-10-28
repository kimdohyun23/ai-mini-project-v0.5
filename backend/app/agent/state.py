"""
그래프 상태(State)
──────────────────────────────────────────────────────────────
모든 노드가 함께 읽고 쓰는 공유 데이터입니다.
노드는 상태 전체를 돌려주지 않고, "바꾸고 싶은 항목만" 담은 dict를 돌려줍니다.
LangGraph가 그 dict를 기존 상태에 반영합니다.
"""
from __future__ import annotations

from typing import Annotated, TypedDict


def merge_findings(current: list[dict], incoming: list[dict]) -> list[dict]:
    """회사별 결과를 병합하고 같은 회사의 재전달은 덮어써 중복을 막습니다."""
    merged = {entry["competitor"]: entry for entry in current}
    merged.update({entry["competitor"]: entry for entry in incoming})
    return list(merged.values())


class AgentState(TypedDict, total=False):
    # ── 입력 ─────────────────────────────
    run_id: str
    item_id: str
    item_name: str
    competitors: list[str]      # 분석 대상 기업 (items.yaml)
    query: str                  # 사용자 질문
    test_mode: bool             # 근거/기업 분석/보고서에 부실 결과를 주입하는 점검 모드

    # ── Supervisor ──────────────────────
    intent: str                # overview / comparison / latest
    research_scope: str        # docs / web / all
    research_count: int        # 재검색 상한을 위한 검색 노드 실행 횟수
    web_research_count: int     # 최신 정보용 웹 검색 수행 횟수
    competitor: str            # Send로 전달한 단일 기업
    analysis_done: bool         # 기업 분석 완료 여부
    routing_history: list[dict] # 상태에 따른 판단과 이유
    next: str                   # 다음에 실행할 노드 이름

    # ── 작업 결과 ────────────────────────
    evidence: list[dict]                # researcher 가 모은 근거 자료
    competitor_findings: Annotated[list[dict], merge_findings]     # competitor_analyst 의 경쟁사별 분석 결과
    draft: str                          # writer 가 작성한 보고서 초안

    # ── 평가 ─────────────────────────────
    eval_passed: bool
    eval_reason: str
    eval_attempts: int

    # DAY 3: 단계별 평가 결과와 재시도 피드백
    stage_attempts: dict[str, int]
    stage_results: dict[str, dict]
    stage_feedback: dict[str, str]
    retry_companies: list[str]
    final_evaluations: list[dict]
    judge_audit: dict
