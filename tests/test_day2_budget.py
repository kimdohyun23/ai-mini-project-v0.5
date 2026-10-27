import asyncio
import json
import unittest
from dataclasses import replace
from unittest.mock import AsyncMock, Mock, patch
import httpx
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from app import llm
from app import main
from app.budget import BudgetExceeded, RunBudget, current_budget, partial_report


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_reservations_never_exceed_limit(self):
        budget = RunBudget(0.0001)

        async def reserve():
            try:
                return await budget.reserve("gpt-4o-mini", 100, 50)
            except BudgetExceeded:
                return None

        reservations = await asyncio.gather(*(reserve() for _ in range(3)))
        self.assertEqual(sum(x is not None for x in reservations), 2)
        self.assertLessEqual(budget.spent + budget.reserved, budget.limit)

    async def test_normal_call_settles_actual_usage(self):
        budget = RunBudget(0.01)
        token = current_budget.set(budget)
        provider = Mock()
        provider.bind.return_value = provider
        provider.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )
        )
        try:
            with patch.object(llm, "get_llm", return_value=provider):
                result = await llm.ainvoke_tracked(
                    [HumanMessage(content="test")], node="test", max_tokens=100
                )
            self.assertEqual(result, "ok")
            self.assertAlmostEqual(float(budget.spent), 0.0000045)
            self.assertEqual(budget.reserved, 0)
            provider.bind.assert_called_once_with(max_tokens=100)
        finally:
            current_budget.reset(token)

    async def test_unexpected_usage_overrun_logs_reason_and_stops_last_call(self):
        budget = RunBudget(0.001)
        token = current_budget.set(budget)
        provider = Mock()
        provider.bind.return_value = provider
        provider.ainvoke = AsyncMock(
            return_value=AIMessage(
                content="provider response",
                usage_metadata={
                    "input_tokens": 10000,
                    "output_tokens": 5,
                    "total_tokens": 10005,
                },
            )
        )
        events = []
        try:
            with (
                patch.object(llm, "get_llm", return_value=provider),
                patch.object(
                    llm,
                    "emit",
                    side_effect=lambda event: events.append(event.model_dump()),
                ),
            ):
                with self.assertRaisesRegex(
                    BudgetExceeded, "공급자가 반환한 토큰 사용량"
                ):
                    await llm.ainvoke_tracked(
                        [HumanMessage(content="test")], node="writer", max_tokens=100
                    )
            self.assertTrue(budget.blocked)
            self.assertGreater(budget.spent, budget.limit)
            self.assertEqual(budget.reserved, 0)
            self.assertEqual(len([e for e in events if e["type"] == "llm_usage"]), 1)
            self.assertTrue(
                any(
                    e["type"] == "progress" and "상한 초과로 중단" in e["message"]
                    for e in events
                )
            )
        finally:
            current_budget.reset(token)

    async def test_denied_call_never_reaches_provider(self):
        budget = RunBudget(0.000001)
        token = current_budget.set(budget)
        try:
            with patch.object(llm, "get_llm") as provider:
                with self.assertRaises(BudgetExceeded):
                    await llm.ainvoke_tracked(
                        [HumanMessage(content="test")], node="test"
                    )
                provider.assert_not_called()
            self.assertEqual(budget.spent, 0)
        finally:
            current_budget.reset(token)

    async def test_provider_failure_preserves_conservative_charge(self):
        budget = RunBudget(0.01)
        token = current_budget.set(budget)
        provider = Mock()
        provider.bind.return_value = provider
        provider.ainvoke = AsyncMock(side_effect=RuntimeError("controlled failure"))
        try:
            with patch.object(llm, "get_llm", return_value=provider):
                with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                    await llm.ainvoke_tracked(
                        [HumanMessage(content="test")], node="test", max_tokens=100
                    )
            self.assertTrue(budget.blocked)
            self.assertGreater(budget.spent, 0)
            self.assertEqual(budget.reserved, 0)
        finally:
            current_budget.reset(token)

    async def test_streaming_settles_usage_and_emits_tokens(self):
        budget = RunBudget(0.01)
        token = current_budget.set(budget)
        provider = Mock()
        provider.bind.return_value = provider

        async def stream(messages):
            yield AIMessageChunk(
                content="ok",
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )

        provider.astream = stream
        events = []
        try:
            with (
                patch.object(llm, "get_llm", return_value=provider),
                patch.object(
                    llm, "emit", side_effect=lambda e: events.append(e.model_dump())
                ),
            ):
                self.assertEqual(
                    await llm.astream_tracked(
                        [HumanMessage(content="test")], node="test"
                    ),
                    "ok",
                )
            self.assertEqual(len([e for e in events if e["type"] == "token"]), 1)
            self.assertEqual(budget.reserved, 0)
        finally:
            current_budget.reset(token)

    async def test_request_scoped_budgets_do_not_share_spend(self):
        async def run(limit):
            budget = RunBudget(limit)
            token = current_budget.set(budget)
            try:
                try:
                    reservation = await budget.reserve("gpt-4o-mini", 100, 50)
                    await asyncio.sleep(0)
                    await current_budget.get().settle(reservation, 0.00003)
                except BudgetExceeded:
                    pass
                return budget
            finally:
                current_budget.reset(token)

        low, normal = await asyncio.gather(run(0.000001), run(0.01))
        self.assertEqual(float(low.spent), 0)
        self.assertAlmostEqual(float(normal.spent), 0.00003)
        self.assertFalse(normal.blocked)

    async def test_unknown_model_is_blocked_instead_of_zero_cost(self):
        with self.assertRaisesRegex(ValueError, "단가표"):
            await RunBudget(0.01).reserve("unpriced-model", 100, 100)

    async def test_low_budget_api_returns_done_and_incomplete_report(self):
        with (
            patch.object(
                main, "settings", replace(main.settings, llm_cost_limit_usd=0.000001)
            ),
            patch.object(llm, "get_llm") as provider,
        ):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=main.app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/api/chat",
                    json={"item_id": "hbf", "query": "HBF 설명", "test_mode": False},
                )
            provider.assert_not_called()
        events = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertEqual(response.status_code, 200)
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["llm_calls"], 0)
        report = next(e for e in events if e["type"] == "report")
        self.assertIn("완료된 기술 분석 보고서가 아닙니다", report["markdown"])
        self.assertEqual(events[-1]["total_cost_usd"], 0)

    def test_partial_report_retains_existing_evidence_and_company_results(self):
        text = partial_report(
            {
                "evidence": [{"source": "paper.pdf", "content": "근거 문단"}],
                "competitor_findings": [{"competitor": "A", "analysis": "완료 분석"}],
            },
            "budget stop",
        )
        for required in ["paper.pdf", "근거 문단", "완료 분석", "budget stop"]:
            self.assertIn(required, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
