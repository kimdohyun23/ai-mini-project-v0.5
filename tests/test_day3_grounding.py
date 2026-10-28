"""인용 표기와 Judge 근거 선택의 회귀 검증."""

import json
import unittest
from unittest.mock import AsyncMock, Mock, patch

from app import llm
from app.budget import BudgetExceeded, RunBudget, current_budget
from app.agent import evaluation as ev
from app.agent.prompts import attach_references


class CitationFormatTests(unittest.TestCase):
    def test_grouped_and_adjacent_citations(self):
        self.assertEqual(ev.citation_ids("주장 [1, 2][3] [2]"), [1, 2, 3, 2])
        self.assertEqual(ev.citation_ids("날짜 [2026-10] 또는 [1, x]"), [])

    def test_group_remapping_preserves_invalid_local_number(self):
        evidence = [dict(source="a", content="A"), dict(source="b", content="B")]
        registry, rows = ev.globalize_findings(
            evidence, [dict(analysis="주장 [1, 2, 9]", evidence=evidence[::-1])]
        )
        self.assertEqual(rows[0]["analysis"], "주장 [2][1][0]")
        self.assertEqual(ev.globalize_findings(registry, rows)[1], rows)

    def test_grouped_references_are_attached(self):
        result = attach_references(
            "주장 [1, 2]", [dict(title="A", source="a"), dict(title="B", source="b")]
        )
        self.assertIn("[1] A | a", result)
        self.assertIn("[2] B | b", result)

    def test_grouped_invalid_reference_fails_company_check(self):
        self.assertIn("범위 밖 인용 [3]", ev.check_company("TRL: 3 [1, 3]", 2))

    def test_catalog_preserves_visible_evidence_and_identifiers(self):
        content = (
            "An actual paragraph\twith whitespace.\n다음 문장에 근거가 있습니다. " * 50
        )
        catalog = ev.evidence_catalog([dict(source="paper", content=content)])
        self.assertEqual(
            "".join(s["text"] for s in catalog[0]["spans"]), content[:1200]
        )
        self.assertEqual(catalog[0]["spans"][0]["id"], "E1:S1")
        self.assertEqual(
            len({s["id"] for s in catalog[0]["spans"]}), len(catalog[0]["spans"])
        )


class JudgeGroundingTests(unittest.IsolatedAsyncioTestCase):
    async def test_schema_is_forwarded_with_exact_claim_keys(self):
        schema = ev.judge_response_schema([{"id": 2}, {"id": 7}])
        claim_keys = schema["schema"]["properties"]["claims"]
        self.assertEqual(claim_keys["required"], ["2", "7"])
        self.assertFalse(claim_keys["additionalProperties"])
        self.assertEqual(claim_keys["properties"]["2"], {"$ref": "#/$defs/claim"})
        self.assertIn("claim", schema["schema"]["$defs"])
        budget = RunBudget(0.01)
        token = current_budget.set(budget)
        provider = AsyncMock()
        provider.bind = Mock(return_value=provider)
        from langchain_core.messages import AIMessage, HumanMessage

        provider.ainvoke.return_value = AIMessage(
            content="{}",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )
        try:
            with patch.object(llm, "get_llm", return_value=provider):
                await llm.ainvoke_tracked(
                    [HumanMessage(content="test")], node="judge", response_schema=schema
                )
            self.assertEqual(
                provider.bind.call_args.kwargs["response_format"],
                {"type": "json_schema", "json_schema": schema},
            )
            self.assertEqual(budget.reserved, 0)
        finally:
            current_budget.reset(token)

    async def test_schema_input_cost_is_reserved_before_provider_call(self):
        budget = RunBudget(0.0001)
        token = current_budget.set(budget)
        try:
            with patch.object(llm, "get_llm") as provider:
                with self.assertRaises(BudgetExceeded):
                    await llm.ainvoke_tracked(
                        [],
                        node="judge",
                        max_tokens=1,
                        response_schema={"description": "x" * 10000},
                    )
                provider.assert_not_called()
            self.assertEqual(budget.spent, 0)
        finally:
            current_budget.reset(token)

    async def judge(self, *, refs, citations=(1,), supported=True, cited=True):
        state = dict(
            query="용량 설명",
            draft="용량을 지원합니다.[1]",
            evidence=[
                dict(
                    source="paper-a",
                    content="High Bandwidth Flash supports large memory capacity.",
                ),
                dict(source="paper-b", content="A different source about latency."),
            ],
        )
        claims = [dict(id=1, text=state["draft"], citations=list(citations))]
        response = dict(
            relevant=True,
            relevance_reason="직접 답함",
            uncertainty=False,
            uncertainty_reason="한계 없음",
            claims=[
                dict(
                    id=1,
                    supported=supported,
                    cited_support=cited,
                    reason="해당 발췌로 판단",
                    supporting_refs=refs,
                )
            ],
        )
        response["claims"] = {str(r.pop("id")): r for r in response["claims"]}
        with patch.object(
            ev, "ainvoke_tracked", AsyncMock(return_value=json.dumps(response))
        ) as llm:
            verdict = await ev.judge_report(state, claims)
        return verdict.claims[0], json.loads(llm.call_args.args[0][1].content)

    async def test_existing_span_passes_without_model_copying_quote(self):
        row, payload = await self.judge(refs=["E1:S1"])
        self.assertTrue(row.supported)
        self.assertTrue(row.cited_support)
        self.assertIsInstance(payload["evidence"], list)
        self.assertEqual(
            payload["evidence"][0]["spans"][0]["text"],
            "High Bandwidth Flash supports large memory capacity.",
        )

    async def test_missing_source_or_span_never_passes(self):
        for refs in [[], ["E99:S1"], ["E1:S99"], ["E1:S1", "E99:S1"]]:
            with self.subTest(refs=refs):
                row, _ = await self.judge(refs=refs)
                self.assertFalse(row.supported)
                self.assertFalse(row.cited_support)

    async def test_actual_other_source_supports_faithfulness_but_not_citation(self):
        row, _ = await self.judge(refs=["E2:S1"])
        self.assertTrue(row.supported)
        self.assertFalse(row.cited_support)

    async def test_support_can_be_subset_of_multiple_citations(self):
        row, _ = await self.judge(refs=["E1:S1"], citations=(1, 2))
        self.assertTrue(row.cited_support)

    async def test_unsupported_semantics_not_overridden_by_valid_span(self):
        row, _ = await self.judge(refs=["E1:S1"], supported=False)
        self.assertFalse(row.supported)
        self.assertFalse(row.cited_support)

    async def test_invalid_citation_is_not_hidden_by_valid_support(self):
        row, _ = await self.judge(refs=["E1:S1"], citations=(1, 99))
        self.assertFalse(row.cited_support)

    async def test_citation_without_any_number_does_not_pass(self):
        row, _ = await self.judge(refs=["E1:S1"], citations=())
        self.assertFalse(row.cited_support)


if __name__ == "__main__":
    unittest.main()
