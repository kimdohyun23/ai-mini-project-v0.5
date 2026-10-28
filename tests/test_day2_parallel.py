import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from app.agent import graph as graph_module
from app.agent import nodes, supervisor as supervisor_module
from app.agent.state import merge_findings


# DAY3 평가 내용은 test_day3_evaluation에서 검증한다. 이 테스트는 DAY2의 경로/동시성 회귀를 격리한다.
async def pass_intermediate(state):
    return {}


async def pass_final(state):
    return {"eval_passed": True, "eval_attempts": 1}


BASE = {
    "item_id": "hbf",
    "item_name": "HBF",
    "competitors": ["SK hynix", "SanDisk", "Samsung"],
    "query": "세 기업 TRL 비교",
}


class ParallelTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, behavior):
        active = 0
        max_active = 0
        started = asyncio.Event()
        count = 0

        async def analyze(state, company):
            nonlocal active, max_active, count
            active += 1
            max_active = max(max_active, active)
            count += 1
            if count == 3:
                started.set()
            try:
                await started.wait()
                if behavior == "failure" and company == "Samsung":
                    raise ValueError("controlled failure")
                if behavior == "timeout" and company == "Samsung":
                    await asyncio.Event().wait()
                return {
                    "competitor": company,
                    "analysis": "근거 기반 분석",
                    "sources": ["evidence"],
                }
            finally:
                active -= 1

        calls = [
            json.dumps(
                {
                    "intent": "comparison",
                    "next": n,
                    "research_scope": "all",
                    "reason": "상태 확인",
                }
            )
            for n in ["researcher", "competitor_analyst", "writer"]
        ]

        async def research(state):
            return {
                "research_count": 1,
                "web_research_count": 1,
                "evidence": [{"source": "test", "content": "근거"}],
            }

        async def writer(state):
            self.assertEqual(len(state["competitor_findings"]), 3)
            return {"draft": "검증 문장" * 100}

        events = []
        final = {}
        with (
            patch.object(graph_module, "research_evaluator", pass_intermediate),
            patch.object(graph_module, "analysis_evaluator", pass_intermediate),
            patch.object(graph_module, "step_evaluator", pass_final),
            patch.object(
                supervisor_module, "ainvoke_tracked", AsyncMock(side_effect=calls)
            ),
            patch.object(graph_module, "researcher", research),
            patch.object(graph_module, "writer", writer),
            patch.object(nodes, "analyze_competitor", analyze),
            patch.object(
                nodes, "settings", SimpleNamespace(competitor_timeout_seconds=0.1)
            ),
        ):
            async for mode, payload in graph_module.build_graph().astream(
                BASE, stream_mode=["custom", "values"], config={"recursion_limit": 25}
            ):
                if mode == "custom":
                    events.append(payload)
                else:
                    final = payload
        self.assertEqual(max_active, 3)
        joins = [
            e
            for e in events
            if e.get("type") == "node_start" and e.get("node") == "competitor_join"
        ]
        self.assertEqual(len(joins), 1)
        self.assertEqual(len([e for e in events if e.get("type") == "branch_end"]), 3)
        return final

    async def test_three_concurrent_branches_merge_before_writer(self):
        state = await self.exercise("success")
        self.assertEqual(
            {f["competitor"] for f in state["competitor_findings"]},
            set(BASE["competitors"]),
        )
        self.assertTrue(
            all(f["status"] == "success" for f in state["competitor_findings"])
        )

    async def test_failed_branch_preserves_other_companies(self):
        state = await self.exercise("failure")
        self.assertEqual(
            sum(f["status"] == "success" for f in state["competitor_findings"]), 2
        )
        self.assertEqual(
            [
                f["competitor"]
                for f in state["competitor_findings"]
                if f["status"] == "failed"
            ],
            ["Samsung"],
        )

    async def test_timed_out_branch_preserves_other_companies(self):
        state = await self.exercise("timeout")
        self.assertEqual(
            sum(f["status"] == "success" for f in state["competitor_findings"]), 2
        )
        self.assertEqual(
            next(
                f for f in state["competitor_findings"] if f["competitor"] == "Samsung"
            )["error"],
            "timeout",
        )

    async def test_empty_company_list_joins_without_hanging(self):
        self.assertEqual(
            nodes.fan_out_competitors({"competitors": []}), "competitor_join"
        )

    def test_reducer_is_idempotent_for_duplicate_company(self):
        before = [
            {"competitor": "A", "analysis": "old"},
            {"competitor": "B", "analysis": "keep"},
        ]
        after = merge_findings(before, [{"competitor": "A", "analysis": "new"}])
        self.assertEqual(len(after), 2)
        self.assertEqual(after[0]["analysis"], "new")
        self.assertEqual(before[0]["analysis"], "old")


if __name__ == "__main__":
    unittest.main(verbosity=2)
