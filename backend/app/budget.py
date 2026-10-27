"""실행별 LLM 예상 비용과 진행 중인 병렬 호출의 예약 비용을 함께 관리합니다."""
from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal

from app.config import PRICING_PER_1M


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Reservation:
    amount: Decimal
    input_bound: int
    output_limit: int


class RunBudget:
    def __init__(self, limit: float):
        self.limit = Decimal(str(limit))
        if not self.limit.is_finite() or self.limit < 0:
            raise ValueError("LLM_COST_LIMIT_USD는 0 이상의 유한한 값이어야 합니다.")
        self.spent = Decimal("0")
        self.reserved = Decimal("0")
        self.blocked = False
        self.lock = asyncio.Lock()

    async def reserve(self, model: str, input_bound: int, output_limit: int) -> Reservation:
        if model not in PRICING_PER_1M:
            raise ValueError(f"단가표에 없는 모델은 예산을 계산할 수 없습니다: {model}")
        if output_limit <= 0:
            raise ValueError("출력 토큰 상한은 양수여야 합니다.")
        price_in, price_out = [Decimal(str(p)) for p in PRICING_PER_1M[model]]
        amount = (input_bound * price_in + output_limit * price_out) / Decimal(1_000_000)
        async with self.lock:
            if self.blocked or self.spent + self.reserved + amount > self.limit:
                self.blocked = True
                raise BudgetExceeded(
                    f"LLM 비용 상한으로 호출 전 중단: 누적=${self.spent:.6f}, "
                    f"진행 중 예약=${self.reserved:.6f}, 다음 호출 예약=${amount:.6f}, 상한=${self.limit:.6f}"
                )
            self.reserved += amount
        return Reservation(amount, input_bound, output_limit)

    async def settle(self, reservation: Reservation, cost: float) -> None:
        async with self.lock:
            self.reserved -= reservation.amount
            self.spent += Decimal(str(cost))
            if self.spent + self.reserved > self.limit:
                self.blocked = True


current_budget: ContextVar[RunBudget | None] = ContextVar("llm_run_budget", default=None)


def partial_report(state: dict, reason: str) -> str:
    lines = ["# 비용 상한으로 실행 중단", reason, "", "이 결과는 완료된 기술 분석 보고서가 아닙니다."]
    if state.get("draft"):
        lines += ["", "## 중단 전에 작성된 초안", state["draft"]]
    if state.get("competitor_findings"):
        lines += ["", "## 중단 전 수집한 기업별 분석"]
        for finding in state["competitor_findings"]:
            lines += [f"### {finding['competitor']}", finding.get("analysis", "분석 미완료")]
    if state.get("evidence"):
        lines += ["", "## 중단 전 수집한 근거"]
        for i, entry in enumerate(state["evidence"], 1):
            lines += [f"[{i}] {entry.get('title', '')} | {entry.get('source', '')}", entry.get("content", "")[:400]]
    return "\n\n".join(lines)
