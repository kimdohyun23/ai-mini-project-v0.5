import json
import unittest
from unittest.mock import AsyncMock, patch
from app.agent import graph as graph_module
from app.agent import nodes, supervisor as supervisor_module


# DAY3 평가 내용은 test_day3_evaluation에서 검증한다. 이 테스트는 DAY2의 경로/동시성 회귀를 격리한다.
async def pass_intermediate(state):
    return {}


async def pass_final(state):
    return {"eval_passed": True, "eval_attempts": 1}


BASE = {
    "item_id": "hbf",
    "item_name": "HBF",
    "competitors": ["SK hynix", "SanDisk", "Samsung"],
}


def decision(intent, target, scope="docs"):
    return json.dumps(
        {
            "intent": intent,
            "next": target,
            "research_scope": scope,
            "reason": "질문과 현재 상태를 확인함",
        }
    )


class DynamicRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def run_graph(self, query, decisions):
        scopes = []

        async def research(state):
            scope = state["research_scope"]
            scopes.append(scope)
            return {
                "evidence": state.get("evidence", [])
                + [{"source": scope, "content": scope}],
                "research_count": state.get("research_count", 0) + 1,
                "web_research_count": state.get("web_research_count", 0)
                + int(scope in ("web", "all")),
            }

        async def analyze(state):
            return {
                "analysis_done": True,
                "competitor_findings": [
                    {"competitor": c} for c in state["competitors"]
                ],
            }

        async def writer(state):
            return {"draft": "검증용 문장" * 100}

        with (
            patch.object(graph_module, "research_evaluator", pass_intermediate),
            patch.object(graph_module, "analysis_evaluator", pass_intermediate),
            patch.object(graph_module, "step_evaluator", pass_final),
            patch.object(
                supervisor_module, "ainvoke_tracked", AsyncMock(side_effect=decisions)
            ),
            patch.object(graph_module, "researcher", research),
            patch.object(graph_module, "competitor_analyst", analyze),
            patch.object(
                graph_module, "fan_out_competitors", lambda state: "competitor_join"
            ),
            patch.object(graph_module, "writer", writer),
        ):
            result = await graph_module.build_graph().ainvoke(
                {**BASE, "query": query}, {"recursion_limit": 25}
            )
        return result, scopes

    async def test_overview_skips_company_analysis_and_returns_to_supervisor(self):
        state, scopes = await self.run_graph(
            "HBF는 HBM과 무엇이 다른가?",
            [decision("overview", "researcher"), decision("overview", "writer")],
        )
        self.assertEqual(
            [x["target"] for x in state["routing_history"]],
            ["researcher", "writer", "end"],
        )
        self.assertFalse(state.get("competitor_findings"))
        self.assertEqual(scopes, ["docs"])
        self.assertTrue(state["eval_passed"])

    async def test_comparison_requires_company_analysis(self):
        state, scopes = await self.run_graph(
            "세 기업의 TRL을 비교해줘",
            [
                decision("comparison", "researcher", "all"),
                decision("comparison", "competitor_analyst"),
                decision("comparison", "writer"),
            ],
        )
        self.assertEqual(
            [x["target"] for x in state["routing_history"]],
            ["researcher", "competitor_analyst", "writer", "end"],
        )
        self.assertEqual(len(state["competitor_findings"]), 3)
        self.assertEqual(scopes, ["all"])

    async def test_latest_researches_again_and_retains_first_evidence(self):
        state, scopes = await self.run_graph(
            "새로 발표된 고객사 소식이 있어?",
            [
                decision("latest", "researcher"),
                decision("latest", "researcher", "web"),
                decision("latest", "writer"),
            ],
        )
        self.assertEqual(
            [x["target"] for x in state["routing_history"]],
            ["researcher", "researcher", "writer", "end"],
        )
        self.assertEqual(scopes, ["docs", "web"])
        self.assertEqual(state["research_count"], 2)
        self.assertEqual(len(state["evidence"]), 2)

    async def test_illegal_early_writer_is_rejected(self):
        with patch.object(
            supervisor_module,
            "ainvoke_tracked",
            AsyncMock(return_value=decision("overview", "writer")),
        ):
            with self.assertRaisesRegex(ValueError, "허용되지 않은"):
                await supervisor_module.supervisor({**BASE, "query": "설명"})

    async def test_research_upper_bound_is_enforced(self):
        state = {
            **BASE,
            "query": "설명",
            "intent": "overview",
            "research_count": 2,
            "evidence": [],
        }
        self.assertEqual(supervisor_module.allowed_actions(state), ["writer"])

    async def test_writer_retry_limit_does_not_report_pass(self):
        result = await supervisor_module.supervisor(
            {
                **BASE,
                "query": "설명",
                "draft": "짧음",
                "eval_passed": False,
                "eval_attempts": 2,
            }
        )
        self.assertEqual(result["next"], "end")
        self.assertIn("미통과", result["routing_history"][-1]["reason"])

    async def test_search_merges_distinct_chunks_from_same_document(self):
        first = {"source": "same.pdf", "content": "문단 1"}
        second = {"source": "same.pdf", "content": "문단 2"}
        with (
            patch.object(nodes, "index_exists", return_value=True),
            patch.object(nodes, "search_docs", AsyncMock(return_value=[first, second])),
        ):
            result = await nodes.researcher(
                {**BASE, "query": "설명", "research_scope": "docs", "evidence": [first]}
            )
        self.assertEqual(result["evidence"], [first, second])

    async def test_missing_web_key_is_counted_as_attempt_without_fabricated_results(
        self,
    ):
        with patch.object(nodes, "web_search_enabled", return_value=False):
            result = await nodes.researcher(
                {**BASE, "query": "최신", "research_scope": "web"}
            )
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["research_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
