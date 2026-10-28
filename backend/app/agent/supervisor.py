"""질문 의도와 작업 결과를 보고 다음 작업과 종료를 결정합니다."""
from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict

from app.agent.evaluation import MAX_STAGE_ATTEMPTS
from app.agent.state import AgentState
from app.agent.tracing import traced
from app.events import Progress, Route, emit
from app.llm import ainvoke_tracked

MAX_RESEARCH_ROUNDS = 2
MAX_WRITER_ATTEMPTS = 2


class RoutingDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    intent: Literal["overview", "comparison", "latest"]
    next: Literal["researcher", "competitor_analyst", "writer"]
    research_scope: Literal["docs", "web", "all"]
    reason: str


ROUTING_SYSTEM = """당신은 기술 분석 작업의 Supervisor입니다.
고정된 순서표 대신 사용자 질문과 현재 상태를 보고 다음 작업을 선택하세요.
질문 의도: overview=기술 개념/차이 설명, comparison=기업별 현황/TRL 비교,
latest=최근 발표/샘플 출하/고객사 등 최신 소식 확인.
처음 판단한 intent는 이후에도 유지하세요.

작업자:
researcher: 근거 검색. research_scope는 docs(로컬 논문), web(최신 웹), all(둘 다).
competitor_analyst: 기업별 분석. 기업 비교가 필요한 질문에만 선택하세요.
writer: 현재 근거와 분석 결과를 바탕으로 답변 작성.

규칙:
- overview는 로컬 논문을 먼저 검색하고, 근거가 있으면 기업 분석을 생략해 답변하세요.
- comparison은 논문과 웹 근거를 먼저 모으고, 기업 분석 결과가 모인 뒤 답변하세요.
- latest는 먼저 논문에서 기술 배경을 확인한 다음, 별도 web 검색으로 최신 소식을 확인하세요.
  논문 검색 결과만으로 최신 출하 여부를 판단하지 마세요.
- 결과가 없거나 부족하면 researcher를 다시 선택할 수 있지만 최대 2회입니다.
- 같은 기업 분석을 반복하지 마세요. 완료된 작업을 현재 상태에서 확인하세요.
- allowed_actions에 있는 작업만 선택하세요.
JSON 객체만 출력하세요. 키: intent, next, research_scope, reason.
reason에는 실제 질문과 현재 상태에 근거한 짧은 한국어 판단 이유를 쓰세요.
"""


def allowed_actions(state: AgentState) -> list[str]:
    rounds = state.get("research_count", 0)
    if rounds == 0:
        return ["researcher"]
    if state.get("intent") == "latest" and not state.get("web_research_count") and rounds < MAX_RESEARCH_ROUNDS:
        return ["researcher"]
    if not state.get("evidence") and rounds < MAX_RESEARCH_ROUNDS:
        return ["researcher"]
    if state.get("intent") == "comparison" and not state.get("analysis_done") and state.get("competitors"):
        return ["competitor_analyst"]
    actions = ["writer"]
    if rounds < MAX_RESEARCH_ROUNDS:
        actions.append("researcher")
    return actions


def _route(state: AgentState, target: str, reason: str, **update) -> dict:
    emit(Route(source="supervisor", target=target, reason=reason))
    return {
        **update,
        "next": target,
        "routing_history": state.get("routing_history", []) + [{"target": target, "reason": reason}],
    }


@traced("supervisor")
async def supervisor(state: AgentState) -> dict:
    if state.get("draft"):
        if state.get("eval_passed"):
            return _route(state, "end", "작성 결과의 평가 통과를 확인하여 종료")
        if state.get("eval_attempts", 0) >= MAX_WRITER_ATTEMPTS:
            emit(Progress(node="supervisor", message="작성 시도 상한 도달. 평가 미통과 상태로 종료합니다."))
            return _route(state, "end", "작성 2회 상한 도달, 평가 미통과")
        return _route(state, "writer", f"평가 미통과로 재작성: {state.get('eval_reason', '')}")

    # 작성자가 자기 평가하지 않음: 별도 평가 노드 결과를 Supervisor가 보고 재시도를 지시.
    for stage, target in [("researcher", "researcher"), ("competitor_analyst", "competitor_analyst")]:
        result = state.get("stage_results", {}).get(stage)
        attempts = state.get("stage_attempts", {}).get(stage, 0)
        if result and not result["passed"] and attempts < MAX_STAGE_ATTEMPTS:
            return _route(state, target, f"{stage} 평가 미통과, 피드백을 전달해 재시도: {result['reason']}")
    # 중간 평가 상한 이후에도 제한된 결과를 명시하고 최종 평가까지 관찰한다.
    # 테스트 모드에서 두 중간 단계 및 최종 평가를 모두 검증할 수 있다.
    actions = allowed_actions(state)
    summary = {
        "query": state["query"],
        "intent": state.get("intent"),
        "competitors": state.get("competitors", []),
        "research_count": state.get("research_count", 0),
        "web_research_count": state.get("web_research_count", 0),
        "evidence_count": len(state.get("evidence", [])),
        "evidence_preview": [
            {"source": str(e.get("source", "")), "text": str(e.get("content", ""))[:200]}
            for e in state.get("evidence", [])[:8]
        ],
        "analysis_done": state.get("analysis_done", False),
        "stage_results": state.get("stage_results", {}),
        "analyzed_companies": [f.get("competitor") for f in state.get("competitor_findings", [])],
        "allowed_actions": actions,
    }
    raw = await ainvoke_tracked([
        SystemMessage(content=ROUTING_SYSTEM),
        HumanMessage(content=json.dumps(summary, ensure_ascii=False)),
    ], node="supervisor", max_tokens=350, json_mode=True)
    # JSON 형식을 벗어난 판단은 잘못된 경로로 실행하지 않고 API 오류로 명시합니다.
    clean = raw.strip()
    if clean.startswith("```") and clean.endswith("```"):
        clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    decision = RoutingDecision.model_validate_json(clean)
    intent = state.get("intent") or decision.intent
    candidate_state = {**state, "intent": intent}
    actions = allowed_actions(candidate_state)
    if decision.next not in actions:
        raise ValueError(f"Supervisor가 허용되지 않은 작업을 선택했습니다: {decision.next}; 허용={actions}")
    scope = decision.research_scope
    if decision.next == "researcher":
        if intent == "latest":
            scope = "docs" if not state.get("research_count") else "web"
        elif intent == "comparison":
            scope = "all"
    return _route(state, decision.next, decision.reason, intent=intent, research_scope=scope)


def route_from_supervisor(state: AgentState) -> str:
    return state["next"]
