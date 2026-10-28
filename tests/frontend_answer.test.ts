import assert from "node:assert/strict";
import test from "node:test";
import { getCurrentAnswer } from "../frontend/lib/answer";
import type { AgentEvent } from "../frontend/lib/events";

const start = (node = "writer"): AgentEvent => ({ type: "node_start", ts: 0, node });
const token = (text: string, node = "writer"): AgentEvent => ({ type: "token", ts: 0, node, text });
const report = (markdown: string): AgentEvent => ({ type: "report", ts: 0, run_id: "test", markdown });

test("normal stream preserves token order", () => {
  assert.equal(getCurrentAnswer([start(), token("첫 "), token("답변")]), "첫 답변");
});
test("retry replaces the previous draft", () => {
  assert.equal(getCurrentAnswer([start(), token("초안"), start(), token("수정본")]), "수정본");
});
test("retry clears the old draft before its first token", () => {
  assert.equal(getCurrentAnswer([start(), token("초안"), start()]), "");
});
test("other nodes do not clear the writer answer or add their tokens", () => {
  assert.equal(getCurrentAnswer([start(), token("답변"), start("step_evaluator"), token("평가", "final_judge")]), "답변");
});
test("final report includes its status and references without duplication", () => {
  const final = "# 검토 필요\n수정본\n## 근거 목록\n[1] paper.pdf";
  assert.equal(getCurrentAnswer([start(), token("초안"), start(), token("수정본"), report(final)]), final);
});
test("budget stop replaces an already streamed partial answer", () => {
  assert.equal(getCurrentAnswer([start(), token("작성 중"), token("비용 상한 중단"), report("# 비용 상한 중단")]), "# 비용 상한 중단");
});
test("empty input stays empty", () => {
  assert.equal(getCurrentAnswer([]), "");
});
