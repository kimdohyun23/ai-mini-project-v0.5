import re

"""LLM 프롬프트 모음"""

COMPETITOR_ANALYST_SYSTEM = """당신은 반도체 기술 전략 분석가입니다.
주어진 근거 자료만 사용하여 특정 기업의 기술 현황을 분석합니다.
근거 자료에 없는 내용은 추측하지 말고 "근거 부족"이라고 쓰세요.

다음 형식으로 답하세요.
- 현황 요약: (3~5문장)
- TRL 추정: (1~9 중 하나, 또는 "판단 불가"와 근거 부족 이유)
- 추정 근거: (어떤 자료의 어떤 내용 때문인지, 자료 번호 [n] 표시)
- 불확실한 점: (1~2문장)
"""

WRITER_SYSTEM = """당신은 기술 전략 보고서 작성자입니다. 입력한 자료만 사용하세요.
다음 형식의 짧은 Markdown을 작성하세요. 전체 본문은 12개 안팎의 짧은 bullet, 각 bullet은 한 문장입니다.
## 1. 요약
- 질문에 직접 답하는 문장 [n]
## 2. 기술 개요
- 근거로 확인되는 특징 한 문장 [n]
## 3. 경쟁사별 현황
### SK hynix
- 확인된 현황 한 문장 [n]
- TRL 추정: N, 해당 검증 단계의 구체적 근거 [n]
### SanDisk
- 확인된 현황 한 문장 [n]
- TRL 추정: N, 해당 검증 단계의 구체적 근거 [n]
### Samsung
- 확인된 현황 한 문장 [n]
- TRL 추정: N, 해당 검증 단계의 구체적 근거 [n]
## 4. 시사점
- 근거 범위 안에서 한 문장 [n]
## 5. 한계와 불확실성
- 확인된 범위와 자료 한계 한 문장 [n]

규칙:
- 입력의 실제 분석 기업 이름을 ### 제목으로 사용합니다. 위 기업은 예시입니다.
- 모든 사실 문장은 문장별로 끝에 실제 근거 번호 [n]를 붙입니다. 여러 문장 뒤에 번호 한 번만 쓰지 마세요.
- 번호는 입력된 전체 근거 목록과 일치해야 합니다. 새 번호를 만들거나 기업별로 다시 1부터 매기지 마세요.
- TRL 수치는 관련 환경에서 시제품 검증 등 해당 단계의 직접 근거가 있어야 합니다.
- 공동 개발, 사양 발표, 일정, 수상만으로 TRL 5 이상을 단정하지 마세요.
- 수치 TRL을 입증할 자료가 없으면 'TRL 추정: 판단 불가, 검증 환경/시제품 자료 부족'이라고 씁니다.
- 판단 불가에는 사유가 있어야 하며 인용할 자료가 실제 없다면 임의 번호를 붙이지 마세요.
- 일반론적 긍정 전망과 경쟁 순위는 쓰지 않습니다. 본 보고서는 ... 비교합니다 같은 도입문도 생략합니다.
- '## 근거 목록'은 쓰지 마세요. 사용된 번호와 실제 출처는 코드가 뒤에 붙입니다.
- 중간 평가가 미통과라면 그 한계를 밝힙니다. 직전 최종 평가 피드백의 문제 문장을 우선 수정하세요.
"""


def format_evidence(evidence: list[dict]) -> str:
    """근거 자료를 번호를 붙인 텍스트로 바꿉니다. (LLM 입력용)"""
    if not evidence:
        return "(근거 자료 없음)"
    lines = []
    for i, e in enumerate(evidence, start=1):
        lines.append(
            f"[{i}] ({e.get('kind', 'unknown')}) {e.get('title', '')} | {e.get('source', '')}\n{e['content'][:1200]}"
        )
    return "\n\n".join(lines)


def format_findings(findings: list[dict]) -> str:
    if not findings:
        return "(경쟁사 분석 결과 없음)"
    return "\n\n".join(f"### {f['competitor']}\n{f['analysis']}" for f in findings)


def attach_references(draft: str, evidence: list[dict]) -> str:
    """본문 인용은 그대로 두고 실제 사용 번호의 출처만 결정적으로 연결합니다."""
    body = re.split(r"(?m)^##\s+(?:근거 목록|참고.*|출처.*)\n", draft, maxsplit=1)[
        0
    ].rstrip()
    ids = sorted({int(n) for n in re.findall(r"\[(\d+)\]", body)})
    references = [
        f"[{i}] {evidence[i - 1].get('title', '')} | {evidence[i - 1].get('source', '')}"
        for i in ids
        if 1 <= i <= len(evidence)
    ]
    return body + ("\n\n## 근거 목록\n" + "\n".join(references) if references else "")
