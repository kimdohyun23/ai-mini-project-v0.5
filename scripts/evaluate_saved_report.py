"""동일 상태의 보고서를 두 번 독립 평가한다. API 키/실행 결과는 커밋하지 않는다."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
# 이 검증은 로컬 기록만 남긴다. 캡처용 LangSmith 활성화와 별개다.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from app.agent.evaluation import step_evaluator  # noqa: E402
from app.budget import RunBudget, current_budget  # noqa: E402


async def evaluate(args):
    data = args.state.read_bytes()
    state = json.loads(data)
    budget = RunBudget(args.limit)
    token = current_budget.set(budget)
    runs = []
    try:
        for _ in range(2):
            # 동일한 원본 상태를 매번 새로 읽는다. 평가 결과 캐시를 재사용하지 않는다.
            result = await step_evaluator({**state, "eval_attempts": 0})
            runs.append(result)
    finally:
        current_budget.reset(token)
    a, b = [{r["name"]: r for r in run["final_evaluations"]} for run in runs]
    changed = [name for name in a if a[name]["passed"] != b[name]["passed"]]
    result = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "state_sha256": hashlib.sha256(data).hexdigest(),
        "report_sha256": hashlib.sha256(state.get("draft", "").encode()).hexdigest(),
        "same_report_independent_judge_calls": 2,
        "changed_items": changed,
        "changed_count": len(changed),
        "total_items": 5,
        "criterion_passed": len(changed) <= 1,
        "spent_usd": str(budget.spent),
        "cost_limit_usd": args.limit,
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        json.dumps({k: v for k, v in result.items() if k != "runs"}, ensure_ascii=False)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=float, default=0.01)
    asyncio.run(evaluate(parser.parse_args()))
