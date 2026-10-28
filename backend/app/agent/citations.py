"""문장과 기업별 분석에서 공통으로 사용하는 숫자 인용 표기."""

import re
from collections.abc import Callable

CITATION = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")


def citation_ids(text: str) -> list[int]:
    return [
        int(n.strip()) for group in CITATION.findall(text) for n in group.split(",")
    ]


def remap_citations(text: str, resolve: Callable[[int], int]) -> str:
    """[1, 2]와 [1][2]를 같은 방식으로 변환하며 잘못된 번호도 보존한다."""
    return CITATION.sub(
        lambda match: "".join(
            f"[{resolve(int(n.strip()))}]" for n in match.group(1).split(",")
        ),
        text,
    )
