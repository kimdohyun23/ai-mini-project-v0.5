import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agent import evaluation, graph as graph_module, nodes, supervisor


class EvaluationBoundaryTests(unittest.TestCase):
    def test_fractional_and_out_of_range_trl_are_rejected(self):
        for value in ["0", "10", "-3", "3.5", "3.0", "3e1", "3/9", "3~4", "3-4"]:
            with self.subTest(value=value):
                self.assertTrue(evaluation.check_company(f"TRL 추정: {value} [1]", 1))

    def test_integer_trl_remains_valid(self):
        for value in range(1, 10):
            with self.subTest(value=value):
                self.assertEqual(
                    evaluation.check_company(f"TRL 추정: {value} [1]", 1), []
                )
        self.assertEqual(evaluation.check_company("TRL 추정: 3. 개념 연구 [1]", 1), [])

    def test_markdown_bullet_after_trl_is_not_a_numeric_range(self):
        for value in range(1, 10):
            with self.subTest(value=value):
                text = f"- TRL 추정: {value}\n\n- 추정 근거: 개념 검증 자료 [1]"
                self.assertEqual(evaluation.check_company(text, 1), [])
        self.assertTrue(evaluation.check_company("TRL 추정: 3 - 4 [1]", 1))

    def test_empty_unknown_reason_is_rejected(self):
        for text in [
            "TRL 추정: 판단 불가\n추정 근거:",
            "TRL 추정: 판단 불가\n자료:",
            "TRL 추정: 판단 불가\n불확실한 점:",
            "TRL 추정: 판단 불가\n근거 자료 정보",
        ]:
            with self.subTest(text=text):
                self.assertIn("판단 불가 사유 누락", evaluation.check_company(text, 1))

    def test_unknown_with_substantive_missing_evidence_reason_remains_valid(self):
        for text in [
            "TRL 추정: 판단 불가, 자료 부족으로 확인할 수 없음",
            "TRL: 판단 불가\n추정 근거: 공개된 자료에 시제품 검증 결과가 없습니다.",
            "TRL 추정: 판단 불가\n사유: 관련 정보가 공개되지 않았습니다.",
        ]:
            with self.subTest(text=text):
                self.assertEqual(evaluation.check_company(text, 1), [])


class RetryBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_draft_at_cap_ends_without_another_llm_call(self):
        raw = json.dumps(
            dict(
                intent="comparison", next="writer", research_scope="all", reason="작성"
            )
        )
        with patch.object(
            supervisor, "ainvoke_tracked", AsyncMock(return_value=raw)
        ) as llm:
            result = await supervisor.supervisor(
                {
                    "query": "비교",
                    "draft": "",
                    "eval_attempts": 2,
                    "eval_passed": False,
                    "research_count": 2,
                }
            )
        self.assertEqual(result["next"], "end")
        llm.assert_not_awaited()

    async def test_first_empty_draft_retries_writer_without_rerouting_llm(self):
        raw = json.dumps(
            dict(
                intent="comparison", next="writer", research_scope="all", reason="작성"
            )
        )
        with patch.object(
            supervisor, "ainvoke_tracked", AsyncMock(return_value=raw)
        ) as llm:
            result = await supervisor.supervisor(
                {
                    "query": "비교",
                    "draft": "",
                    "eval_attempts": 1,
                    "eval_passed": False,
                    "eval_reason": "빈 보고서",
                    "research_count": 2,
                }
            )
        self.assertEqual(result["next"], "writer")
        self.assertIn("빈 보고서", result["routing_history"][-1]["reason"])
        llm.assert_not_awaited()

    async def test_two_empty_drafts_finish_the_graph_at_the_writer_cap(self):
        companies = ["A", "B", "C"]
        evidence = [
            {
                "kind": "doc",
                "title": str(i),
                "source": f"{i}.pdf",
                "content": "검증 자료",
            }
            for i in range(3)
        ]
        choices = [
            json.dumps(
                dict(intent="comparison", next=n, research_scope="all", reason="검증")
            )
            for n in ["researcher", "competitor_analyst", "writer"]
        ]

        async def analyze(state, company):
            return {
                "competitor": company,
                "analysis": "TRL 추정: 3 [1]",
                "evidence": evidence,
            }

        with (
            patch.object(nodes, "search_docs", AsyncMock(return_value=evidence)),
            patch.object(nodes, "search_web", AsyncMock(return_value=[])),
            patch.object(nodes, "index_exists", return_value=True),
            patch.object(nodes, "web_search_enabled", return_value=True),
            patch.object(nodes, "analyze_competitor", analyze),
            patch.object(
                nodes, "astream_tracked", AsyncMock(return_value="")
            ) as writer,
            patch.object(
                supervisor, "ainvoke_tracked", AsyncMock(side_effect=choices)
            ) as routing,
            patch.object(evaluation, "judge_report", AsyncMock()) as judge,
        ):
            result = await graph_module.build_graph().ainvoke(
                {
                    "item_id": "hbf",
                    "item_name": "HBF",
                    "query": "세 기업 TRL 비교",
                    "competitors": companies,
                },
                config={"recursion_limit": 25},
            )
        self.assertEqual(writer.await_count, 2)
        self.assertEqual(routing.await_count, 3)
        judge.assert_not_awaited()
        self.assertEqual(result["eval_attempts"], 2)
        self.assertEqual(result["next"], "end")
        self.assertFalse(result["eval_passed"])

    async def test_invalid_judge_boolean_is_rejected_with_bounded_panel_messages(self):
        state = {
            "query": "기업 비교",
            "intent": "comparison",
            "competitors": ["A"],
            "draft": nodes.POOR_OUTPUT_SAMPLE,
        }
        claims = evaluation.extract_claims(state["draft"])
        response = {
            "relevant": True,
            "relevance_reason": "관련 있음",
            "uncertainty": True,
            "uncertainty_reason": "한계 있음",
            "claims": [
                {
                    "id": c["id"],
                    "supported": False,
                    "cited_support": [],
                    "reason": "근거 없음",
                }
                for c in claims
            ],
        }
        events = []
        with (
            patch.object(
                evaluation,
                "ainvoke_tracked",
                AsyncMock(return_value=json.dumps(response)),
            ),
            patch.object(
                evaluation, "emit", lambda event: events.append(event.model_dump())
            ),
        ):
            result = await evaluation.step_evaluator(state)
        self.assertFalse(result["eval_passed"])
        self.assertEqual(len(result["judge_audit"]["error_details"]), len(claims))
        for event in events:
            if event.get("stage", "").startswith("final:"):
                self.assertFalse(event["passed"])
                self.assertLess(len(event["reason"]), 400)
                self.assertNotIn("https://errors.pydantic.dev", event["reason"])

    async def test_retry_feedback_logged_once_for_parallel_company_analysis(self):
        state = {
            "item_id": "hbf",
            "item_name": "HBF",
            "competitors": ["A", "B", "C"],
            "stage_feedback": {"competitor_analyst": "근거 번호 누락"},
        }
        events = []
        with (
            patch.object(nodes, "emit", lambda e: events.append(e.model_dump())),
            patch.object(nodes, "search_docs", AsyncMock(return_value=[])),
            patch.object(nodes, "search_web", AsyncMock(return_value=[])),
            patch.object(
                nodes,
                "ainvoke_tracked",
                AsyncMock(return_value="TRL 추정: 판단 불가, 자료 부족"),
            ) as llm,
        ):
            await nodes.competitor_analyst(state)
            await asyncio.gather(
                *(nodes.analyze_competitor(state, c) for c in state["competitors"])
            )
        feedback = [e for e in events if "직전 평가 이유 반영" in e.get("message", "")]
        self.assertEqual(len(feedback), 1)
        self.assertEqual(llm.await_count, 3)
        for call in llm.await_args_list:
            self.assertIn("근거 번호 누락", call.args[0][1].content)


if __name__ == "__main__":
    unittest.main()
