import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from app.agent import evaluation as ev
from app.agent import graph as graph_module
from app.agent import nodes
from app.agent import supervisor as sup
from app.agent.prompts import format_evidence
from app.budget import RunBudget, current_budget

COMPANIES = ["SK hynix", "SanDisk", "Samsung"]
EVIDENCE = [
    dict(kind="doc", title=str(i), source=f"{i}.pdf", content=f"자료 {i}")
    for i in range(3)
]
BASE = dict(
    item_id="hbf",
    item_name="HBF",
    query="세 기업의 HBF와 TRL 비교",
    intent="comparison",
    competitors=COMPANIES,
    evidence=EVIDENCE,
)


def verdict_for(claims, **overrides):
    data = dict(
        relevant=True,
        relevance_reason="질문에 직접 답함",
        uncertainty=True,
        uncertainty_reason="자료 부족 공개",
        claims=[
            dict(
                id=c["id"],
                supported=True,
                cited_support=bool(c["citations"]),
                reason="근거 일치",
            )
            for c in claims
        ],
    )
    data.update(overrides)
    return ev.JudgeVerdict.model_validate(data)


def good_draft():
    return (
        "# 보고서\n## 1. 요약\nHBF는 플래시 기반입니다.[1]\n## 3. 기업별 현황\n"
        + "\n".join(
            f"### {c}\nTRL 추정: 3, 개념 연구 단계입니다.[1]" for c in COMPANIES
        )
        + "\n## 5. 한계와 불확실성\n연구 자료이므로 실제 양산 여부는 확인되지 않습니다.[1]\n"
        + "## 근거 목록\n[1] 0.pdf\n"
    )


class CitationTests(unittest.TestCase):
    def test_local_index_remapped_and_same_document_different_chunks_retained(self):
        a, b = EVIDENCE[:2]
        same_source_new_chunk = {**a, "content": "다른 문단"}
        findings = [
            {
                "competitor": "A",
                "analysis": "사실 [1]과 [2]",
                "evidence": [b, same_source_new_chunk],
            }
        ]
        registry, rows = ev.globalize_findings([a], findings)
        self.assertEqual(len(registry), 3)
        self.assertEqual(rows[0]["analysis"], "사실 [2]과 [3]")
        self.assertEqual(ev.globalize_findings(registry, rows), (registry, rows))
        self.assertEqual(format_evidence(registry).count("[3]"), 1)

    def test_invalid_local_reference_does_not_become_valid_global_reference(self):
        registry, rows = ev.globalize_findings(
            EVIDENCE, [{"analysis": "틀린 [3]", "evidence": EVIDENCE[:1]}]
        )
        self.assertEqual(rows[0]["analysis"], "틀린 [0]")
        self.assertIn(
            "범위 밖", ",".join(ev.check_company("TRL 추정: 3 [0]", len(registry)))
        )

    def test_missing_company_not_hidden_by_another_company_trl(self):
        text = "### SK hynix\nTRL 추정: 4 [1]\n### Samsung\n근거 부족\n"
        sections = ev.company_sections(text, COMPANIES)
        self.assertNotIn("SanDisk", sections)
        self.assertTrue(ev.check_company(sections["Samsung"], 3))

    def test_unknown_requires_reason_and_cannot_also_claim_numeric_trl(self):
        self.assertEqual(
            ev.check_company("TRL 추정: 판단 불가, 자료 부족으로 확인할 수 없음", 0), []
        )
        self.assertTrue(ev.check_company("TRL: 판단 불가", 0))
        self.assertTrue(ev.check_company("TRL: 8, 판단 불가, 근거 부족", 0))

    def test_valid_number_but_wrong_source_fails_citation_accuracy(self):
        state = {**BASE, "draft": good_draft()}
        claims = ev.extract_claims(state["draft"])
        verdict = verdict_for(claims)
        verdict.claims[0].cited_support = False
        results = {r["name"]: r for r in ev.final_results(state, claims, verdict)}
        self.assertTrue(results["Faithfulness"]["passed"])
        self.assertFalse(results["Citation Accuracy"]["passed"])
        self.assertTrue(results["Completeness"]["passed"])

    def test_faithfulness_threshold_boundary_and_missing_citation(self):
        claims = [dict(id=i + 1, text="사실", citations=[1]) for i in range(5)]
        verdict = verdict_for(claims)
        verdict.claims[0].supported = False
        results = {r["name"]: r for r in ev.final_results(BASE, claims, verdict)}
        self.assertTrue(results["Faithfulness"]["passed"])  # 4/5 = 0.8
        verdict.claims[1].supported = False
        results = {r["name"]: r for r in ev.final_results(BASE, claims, verdict)}
        self.assertFalse(results["Faithfulness"]["passed"])
        claims[0]["citations"] = []
        results = {r["name"]: r for r in ev.final_results(BASE, claims, verdict)}
        self.assertFalse(results["Citation Accuracy"]["passed"])

    def test_references_do_not_inflate_claim_denominator(self):
        claims = ev.extract_claims(good_draft())
        self.assertEqual(len(claims), 5)
        self.assertFalse(any(c["text"] == "[1] 0.pdf" for c in claims))

    def test_empty_and_malformed_judge_never_pass_semantics(self):
        result = ev.final_results(BASE, [], None, "Judge 장애")
        self.assertFalse(any(r["passed"] for r in result))
        with self.assertRaises(ValueError):
            ev.JudgeVerdict.model_validate({"relevant": "true"})


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_source_missing_fails_then_cap_visible(self):
        state = {**BASE, "evidence": [{**e, "source": ""} for e in EVIDENCE]}
        events = []
        with patch.object(ev, "emit", lambda e: events.append(e.model_dump())):
            first = await ev.research_evaluator(state)
            second = await ev.research_evaluator({**state, **first})
        self.assertFalse(first["stage_results"]["researcher"]["passed"])
        self.assertTrue(second["stage_results"]["researcher"]["exhausted"])
        self.assertEqual(second["stage_attempts"]["researcher"], 2)
        self.assertIn("상한", events[-2]["reason"])

    async def test_company_gap_failed_and_successful_findings_kept(self):
        rows = [
            dict(competitor=c, analysis="TRL 추정: 3 [1]", status="success")
            for c in COMPANIES
        ]
        rows[1]["analysis"] = "TRL 누락"
        state = {**BASE, "competitor_findings": rows}
        update = await ev.analysis_evaluator(state)
        self.assertEqual(update["retry_companies"], ["SanDisk"])
        self.assertFalse(update["stage_results"]["competitor_analyst"]["passed"])
        routed = await sup.supervisor({**state, **update, "analysis_done": True})
        self.assertEqual(routed["next"], "competitor_analyst")
        self.assertIn("SanDisk", routed["routing_history"][0]["reason"])

    async def test_feedback_in_research_and_writer_input(self):
        state = {
            **BASE,
            "research_scope": "docs",
            "stage_feedback": {"researcher": "출처 누락"},
            "eval_attempts": 1,
            "eval_reason": "인용 오류",
        }
        with (
            patch.object(nodes, "index_exists", return_value=True),
            patch.object(
                nodes, "search_docs", AsyncMock(return_value=EVIDENCE)
            ) as search,
            patch.object(
                nodes, "astream_tracked", AsyncMock(return_value="수정")
            ) as writer,
        ):
            await nodes.researcher(state)
            await nodes.writer(state)
        self.assertIn("출처 누락", search.call_args.args[1])
        self.assertIn("인용 오류", writer.call_args.args[0][1].content)

    async def test_analysis_feedback_in_llm_input(self):
        with (
            patch.object(nodes, "search_docs", AsyncMock(return_value=EVIDENCE)),
            patch.object(nodes, "search_web", AsyncMock(return_value=[])),
            patch.object(
                nodes, "ainvoke_tracked", AsyncMock(return_value="TRL 추정: 3 [1]")
            ) as llm,
        ):
            await nodes.analyze_competitor(
                {**BASE, "stage_feedback": {"competitor_analyst": "Samsung TRL 누락"}},
                "Samsung",
            )
        self.assertIn("Samsung TRL 누락", llm.call_args.args[0][1].content)

    async def test_selected_span_must_come_from_actual_evidence(self):
        evidence = [
            {
                "kind": "doc",
                "title": "논문",
                "source": "paper.pdf",
                "content": "High Bandwidth Flash supports large memory capacity.",
            }
        ]
        claims = [{"id": 1, "text": "대용량을 지원합니다.[1]", "citations": [1]}]
        for reference, expected in [("E1:S1", True), ("E1:S99", False)]:
            data = verdict_for(claims).model_dump()
            data["claims"][0].update(supporting_refs=[reference])
            data["claims"] = {str(r.pop("id")): r for r in data["claims"]}
            with patch.object(
                ev, "ainvoke_tracked", AsyncMock(return_value=json.dumps(data))
            ):
                actual = await ev.judge_report({**BASE, "evidence": evidence}, claims)
            self.assertEqual(actual.claims[0].supported, expected)
            self.assertEqual(actual.claims[0].cited_support, expected)

    async def test_citation_accuracy_denominator_includes_uncited_claims(self):
        claims = [
            {"id": 1, "text": "사실[1]", "citations": [1]},
            {"id": 2, "text": "번호 없는 사실", "citations": []},
        ]
        results = ev.final_results(BASE, claims, verdict_for(claims))
        citation = next(r for r in results if r["name"] == "Citation Accuracy")
        self.assertFalse(citation["passed"])
        self.assertIn("1/2 = 0.500", citation["reason"])

    async def test_judge_rejects_missing_duplicate_and_extra_claim_ids(self):
        claims = ev.extract_claims(good_draft())
        for ids in [[1], [1, 1, 2, 3, 4], [1, 2, 3, 4, 99]]:
            raw = verdict_for(claims).model_dump()
            raw["claims"] = [
                dict(id=i, supported=True, cited_support=True, reason="검증")
                for i in ids
            ]
            raw["claims"] = {str(r.pop("id")): r for r in raw["claims"]}
            with patch.object(
                ev, "ainvoke_tracked", AsyncMock(return_value=json.dumps(raw))
            ):
                with self.assertRaisesRegex(ValueError, "주장 ID"):
                    await ev.judge_report(BASE, claims)

    async def test_final_five_items_and_bad_report_fails_required_items(self):
        state = {**BASE, "draft": nodes.POOR_OUTPUT_SAMPLE}
        claims = ev.extract_claims(state["draft"])
        with patch.object(
            ev, "judge_report", AsyncMock(return_value=verdict_for(claims))
        ):
            result = await ev.step_evaluator(state)
        items = {r["name"]: r for r in result["final_evaluations"]}
        self.assertEqual(len(items), 5)
        self.assertFalse(items["Citation Accuracy"]["passed"])
        self.assertFalse(items["Completeness"]["passed"])
        self.assertFalse(result["eval_passed"])

    async def test_judge_failure_is_fail_closed_and_budget_not_swallowed(self):
        state = {**BASE, "draft": good_draft()}
        with patch.object(
            ev, "judge_report", AsyncMock(side_effect=ValueError("잘못된 JSON"))
        ):
            result = await ev.step_evaluator(state)
        self.assertFalse(result["eval_passed"])
        from app.budget import BudgetExceeded

        with patch.object(
            ev, "judge_report", AsyncMock(side_effect=BudgetExceeded("상한"))
        ):
            with self.assertRaises(BudgetExceeded):
                await ev.step_evaluator(state)

    async def test_test_mode_both_stages_retry_at_cap_and_finish_without_recursion_error(
        self,
    ):
        decisions = [
            json.dumps(
                dict(intent="comparison", next=n, research_scope="all", reason="검증")
            )
            for n in ["researcher", "competitor_analyst", "writer"]
        ]

        async def judge(state, claims):
            return verdict_for(claims, relevant=False, relevance_reason="일반론")

        records, final = [], {}
        with (
            patch.object(sup, "ainvoke_tracked", AsyncMock(side_effect=decisions)),
            patch.object(nodes, "search_docs", AsyncMock(return_value=EVIDENCE)),
            patch.object(nodes, "search_web", AsyncMock(return_value=[])),
            patch.object(nodes, "index_exists", return_value=True),
            patch.object(nodes, "web_search_enabled", return_value=True),
            patch.object(ev, "judge_report", judge),
        ):
            async for mode, payload in graph_module.build_graph().astream(
                {**BASE, "test_mode": True},
                stream_mode=["custom", "values"],
                config={"recursion_limit": 25},
            ):
                if mode == "custom":
                    records.append(payload)
                else:
                    final = payload
        self.assertEqual(
            final["stage_attempts"], {"researcher": 2, "competitor_analyst": 2}
        )
        self.assertEqual(final["eval_attempts"], 2)
        self.assertFalse(final["eval_passed"])
        self.assertTrue(
            any("직전 평가 이유 반영" in e.get("message", "") for e in records)
        )
        self.assertEqual(
            sum(e.get("stage") == "final:Citation Accuracy" for e in records), 2
        )
        self.assertEqual(final["next"], "end")

    async def test_normal_all_three_parallel_results_and_two_stage_evaluations(self):
        active, peak = 0, 0
        barrier = asyncio.Event()

        async def analyze(state, company):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 3:
                barrier.set()
            await barrier.wait()
            active -= 1
            return dict(
                competitor=company,
                analysis="TRL 추정: 3 [1]",
                sources=["0.pdf"],
                evidence=EVIDENCE,
            )

        decisions = [
            json.dumps(
                dict(intent="comparison", next=n, research_scope="all", reason="검증")
            )
            for n in ["researcher", "competitor_analyst", "writer"]
        ]

        async def judge(state, claims):
            return verdict_for(claims)

        with (
            patch.object(sup, "ainvoke_tracked", AsyncMock(side_effect=decisions)),
            patch.object(nodes, "search_docs", AsyncMock(return_value=EVIDENCE)),
            patch.object(nodes, "search_web", AsyncMock(return_value=[])),
            patch.object(nodes, "index_exists", return_value=True),
            patch.object(nodes, "web_search_enabled", return_value=True),
            patch.object(nodes, "analyze_competitor", analyze),
            patch.object(
                nodes, "astream_tracked", AsyncMock(return_value=good_draft())
            ),
            patch.object(ev, "judge_report", judge),
        ):
            final = await graph_module.build_graph().ainvoke(
                BASE, config={"recursion_limit": 25}
            )
        self.assertEqual(peak, 3)
        self.assertEqual(
            final["stage_attempts"], {"researcher": 1, "competitor_analyst": 1}
        )
        self.assertTrue(final["eval_passed"])

    async def test_final_judge_uses_day2_budget_before_provider_call(self):
        from app import llm
        from app.budget import BudgetExceeded

        token = current_budget.set(RunBudget(0.000001))
        provider = AsyncMock()
        try:
            with patch.object(llm, "get_llm", provider):
                with self.assertRaises(BudgetExceeded):
                    await ev.judge_report(BASE, ev.extract_claims(good_draft()))
            provider.assert_not_called()
        finally:
            current_budget.reset(token)


if __name__ == "__main__":
    unittest.main()
