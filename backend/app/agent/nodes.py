"""
작업자(worker) 노드와 평가 노드
──────────────────────────────────────────────────────────────
  researcher          근거 자료 수집
  competitor_analyst  분석 대상 기업별 현황 분석
  writer              보고서 작성 (답변 스트리밍)
  step_evaluator      작성된 보고서 평가
"""
from __future__ import annotations

import asyncio
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from app.agent.evaluation import globalize_findings
from app.agent.prompts import (
    COMPETITOR_ANALYST_SYSTEM,
    WRITER_SYSTEM,
    attach_references,
    format_evidence,
    format_findings,
)
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.config import settings
from app.events import BranchEnd, BranchStart, Progress, Token, emit
from app.llm import ainvoke_tracked, astream_tracked
from app.retrieval import index_exists, search_docs, search_web, web_search_enabled


# ──────────────────────────────────────────────────────────────
# researcher : 근거 자료 수집
# ──────────────────────────────────────────────────────────────
@traced("researcher")
async def researcher(state: AgentState) -> dict:
    item_id, query = state["item_id"], state["query"]
    feedback = state.get("stage_feedback", {}).get("researcher", "")
    if feedback:
        emit(Progress(node="researcher", message=f"재시도 입력에 직전 평가 이유 반영: {feedback}"))
        query += f" 자료 보완 요청: {feedback}"
    scope = state.get("research_scope", "all")
    round_number = state.get("research_count", 0) + 1
    emit(Progress(node="researcher", message=f"검색 {round_number}회차, 범위={scope}"))
    docs, web = [], []
    if scope in ("docs", "all"):
        if not index_exists(item_id):
            emit(Progress(node="researcher", message=f"'{item_id}' 문서 인덱스가 없습니다. scripts/ingest.py 를 먼저 실행하세요."))
        docs = await search_docs(item_id, query, k=4)
        emit(Progress(node="researcher", message=f"문서 검색 {len(docs)}건"))
    if scope in ("web", "all"):
        if web_search_enabled():
            web = await search_web(f"{state['item_name']} {query}", max_results=3)
            emit(Progress(node="researcher", message=f"웹 검색 {len(web)}건"))
        else:
            emit(Progress(node="researcher", message="TAVILY_API_KEY 가 없어 웹 검색을 건너뜁니다. 최신성 확인이 제한됩니다."))
    evidence = list(state.get("evidence", []))
    known = {(str(e.get("source", "")), str(e.get("content", ""))) for e in evidence}
    if state.get("test_mode"):
        emit(Progress(node="researcher", message="[테스트 모드] 실제 검색 결과의 출처를 제거해 중간 평가 실패를 주입합니다."))
        evidence = []
        known = set()
        docs = [{**entry, "source": ""} for entry in docs]
        web = [{**entry, "source": ""} for entry in web]
    for entry in docs + web:
        key = (str(entry.get("source", "")), str(entry.get("content", "")))
        if key not in known:
            evidence.append(entry)
            known.add(key)
    return {
        "evidence": evidence,
        "research_count": round_number,
        "web_research_count": state.get("web_research_count", 0) + int(scope in ("web", "all")),
    }


# ──────────────────────────────────────────────────────────────
# competitor_analyst : 분석 대상 기업별 현황 분석
#  - 기업마다 근거를 추가로 찾고, LLM으로 현황과 TRL을 추정합니다.
#  - 기업 하나의 분석을 "갈래(branch)"라고 부르며, 갈래마다 시작/종료 이벤트를 남깁니다.
# ──────────────────────────────────────────────────────────────
async def analyze_competitor(state: AgentState, competitor: str) -> dict:
    """기업 하나를 분석해 {"competitor": 기업명, "analysis": 분석 내용, "sources": [...]} 를 돌려줍니다."""
    if state.get("test_mode"):
        return {"competitor": competitor, "analysis": "현황 요약: 여러 기업이 기술을 개발하고 있습니다.", "sources": [], "evidence": []}
    item_name = state["item_name"]
    search_query = f"{competitor} {item_name}"

    docs = await search_docs(state["item_id"], search_query, k=3)
    web = await search_web(search_query, max_results=2)
    evidence = docs + web

    feedback = state.get("stage_feedback", {}).get("competitor_analyst", "")
    if feedback:
        emit(Progress(node="competitor_analyst", message=f"{competitor} 재시도 입력에 직전 평가 이유 반영: {feedback}"))
    messages = [
        SystemMessage(content=COMPETITOR_ANALYST_SYSTEM),
        HumanMessage(content=(
            f"분석 아이템: {item_name}\n분석 대상 기업: {competitor}\n\n"
            f"근거 자료:\n{format_evidence(evidence)}\n\n직전 평가 피드백: {feedback or '없음'}"
        )),
    ]
    analysis = await ainvoke_tracked(messages, node="competitor_analyst", max_tokens=650)
    return {
        "competitor": competitor,
        "analysis": analysis,
        "sources": [e["source"] for e in evidence],
        "evidence": evidence,
    }


@traced("competitor_analyst")
async def competitor_analyst(state: AgentState) -> dict:
    competitors = state.get("retry_companies") or state.get("competitors", [])
    feedback = state.get("stage_feedback", {}).get("competitor_analyst", "")
    if feedback:
        emit(Progress(node="competitor_analyst", message=f"재시도 입력에 직전 평가 이유 반영: {feedback}"))
    emit(Progress(node="competitor_analyst", message=f"분석 대상 {len(competitors)}개 기업을 동시에 시작: {', '.join(competitors)}"))
    return {}


def fan_out_competitors(state: AgentState):
    competitors = list(dict.fromkeys(state.get("retry_companies") or state.get("competitors", [])))
    if not competitors:
        return "competitor_join"
    return [Send("competitor_branch", {**state, "competitor": company}) for company in competitors]


@traced("competitor_branch")
async def competitor_branch(state: AgentState) -> dict:
    company = state["competitor"]
    emit(BranchStart(node="competitor_analyst", branch=company))
    started = time.perf_counter()
    try:
        finding = await asyncio.wait_for(
            analyze_competitor(state, company), timeout=settings.competitor_timeout_seconds,
        )
        finding["status"] = "success"
    except Exception as error:
        # 한 갈래의 실패를 데이터로 남겨 다른 기업 결과와 함께 병합합니다.
        kind = "timeout" if isinstance(error, TimeoutError) else type(error).__name__
        emit(Progress(node="competitor_analyst", message=f"{company} 분석 실패 ({kind}). 다른 기업 결과는 보존합니다."))
        finding = {
            "competitor": company,
            "analysis": f"분석 실패 ({kind}). 이 기업의 TRL은 판단할 수 없습니다.",
            "sources": [], "status": "failed", "error": kind,
        }
    finally:
        elapsed = int((time.perf_counter() - started) * 1000)
        emit(BranchEnd(node="competitor_analyst", branch=company, duration_ms=elapsed))
    return {"competitor_findings": [finding]}


@traced("competitor_join")
async def competitor_join(state: AgentState) -> dict:
    findings = state.get("competitor_findings", [])
    succeeded = sum(f.get("status") == "success" for f in findings)
    emit(Progress(node="competitor_analyst", message=f"결과 병합 {len(findings)}/{len(set(state.get('competitors', [])))}개 기업, 성공 {succeeded}, 실패 {len(findings) - succeeded}"))
    evidence, converted = globalize_findings(state.get("evidence", []), findings)
    return {"analysis_done": True, "evidence": evidence, "competitor_findings": converted}


# ──────────────────────────────────────────────────────────────
# writer : 보고서 작성 (작성 중인 내용을 Frontend로 스트리밍)
# ──────────────────────────────────────────────────────────────
POOR_OUTPUT_SAMPLE = (
    "## 1. 요약\n해당 기술은 업계에서 많은 관심을 받고 있으며 향후 성장이 기대됩니다. "
    "여러 기업이 개발을 진행하고 있고 시장 상황에 따라 경쟁이 치열해질 것으로 보입니다.\n\n"
    "## 2. 기술 개요\n이 기술은 성능과 효율을 높이기 위한 차세대 기술입니다. "
    "다양한 분야에서 활용될 수 있으며 중요성이 계속 커지고 있습니다.\n\n"
    "## 3. 경쟁사별 현황\n주요 기업들은 각자의 전략에 따라 기술 개발에 힘쓰고 있습니다. "
    "각 기업의 기술 수준은 상당히 높은 편으로 판단됩니다.\n\n"
    "## 4. 시사점\n지속적인 투자와 연구 개발이 필요합니다.\n\n"
    "## 5. 한계와 불확실성\n시장 상황은 변할 수 있습니다.\n"
)


@traced("writer")
async def writer(state: AgentState) -> dict:
    feedback = state.get("eval_reason", "") if state.get("eval_attempts") else ""
    if feedback:
        emit(Progress(node="writer", message=f"재작성 입력에 직전 평가 이유 반영: {feedback}"))
    # 테스트용 고정 부실 결과에도 피드백이 전달되었음을 기록. 고의 실패 주입은 유지.
    if state.get("test_mode"):
        emit(Progress(node="writer", message="[테스트 모드] 피드백을 수신한 뒤 고정 부실 보고서를 다시 생성합니다."))
        emit(Token(node="writer", text=POOR_OUTPUT_SAMPLE))
        return {"draft": POOR_OUTPUT_SAMPLE}

    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=(
            f"사용자 질문: {state['query']}\n"
            f"분석 아이템: {state['item_name']}\n"
            f"분석 대상 기업: {', '.join(state.get('competitors', []))}\n\n"
            f"근거 자료:\n{format_evidence(state.get('evidence', []))}\n\n"
            f"경쟁사 분석 결과:\n{format_findings(state.get('competitor_findings', []))}\n\n"
            f"중간 단계 평가: {state.get('stage_results', {})}\n"
            f"직전 최종 평가 피드백: {feedback or '없음'}"
        )),
    ]
    draft = await astream_tracked(messages, node="writer")
    return {"draft": attach_references(draft, state.get("evidence", []))}
