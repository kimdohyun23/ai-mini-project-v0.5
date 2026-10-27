"""LLM 호출 전에 비용을 예약하고 사용량과 예상 비용을 기록합니다."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.budget import Reservation, current_budget
from app.config import PRICING_PER_1M, settings
from app.events import LlmUsage, Progress, Token, emit


@lru_cache
def get_llm() -> ChatOpenAI:
    # 자동 재시도를 꺼서 예약 없이 추가 호출이 과금되는 것을 막습니다.
    return ChatOpenAI(model=settings.llm_model, temperature=0, stream_usage=True, max_retries=0, timeout=35)


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICING_PER_1M.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


async def _reserve(messages: list[BaseMessage], output_limit: int) -> Reservation | None:
    budget = current_budget.get()
    if budget is None:
        raise RuntimeError("LLM 호출에는 실행별 RunBudget이 필요합니다.")
    # 현재 API는 텍스트 메시지만 사용합니다. UTF-8 바이트 수에 메시지/역할 여유분을
    # 더해 입력 토큰을 보수적으로 예약하고, 응답 후 사용량 메타데이터로 정산합니다.
    input_bound = 256 + sum(len((message.type + str(message.content)).encode("utf-8")) + 32 for message in messages)
    return await budget.reserve(settings.llm_model, input_bound, output_limit)


async def _record(node: str, message, reservation: Reservation, uncertain: bool = False) -> None:
    budget = current_budget.get()
    usage = getattr(message, "usage_metadata", None) or {}
    if uncertain or "input_tokens" not in usage or "output_tokens" not in usage:
        # 실패 또는 취소된 요청이 실제 과금됐는지는 알 수 없으므로 예약 상한을 차감합니다.
        input_tokens, output_tokens = reservation.input_bound, reservation.output_limit
        budget.blocked = True
        emit(Progress(node=node, message="사용량을 확정할 수 없어 예약 상한으로 보수적 비용을 기록하고 추가 LLM 호출을 중단합니다."))
    else:
        input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
    cost = estimate_cost(settings.llm_model, input_tokens, output_tokens)
    await budget.settle(reservation, cost)
    emit(LlmUsage(node=node, model=settings.llm_model, input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=cost))
    emit(Progress(node=node, message=f"LLM 누적 예상 비용 ${budget.spent:.6f} / 상한 ${budget.limit:.6f}"))


async def ainvoke_tracked(messages: list[BaseMessage], node: str, *, max_tokens: int | None = None, json_mode: bool = False) -> str:
    output_limit = max_tokens or settings.llm_max_output_tokens
    reservation = await _reserve(messages, output_limit)
    options = {"max_tokens": output_limit}
    if json_mode:
        options["response_format"] = {"type": "json_object"}
    response = None
    try:
        response = await get_llm().bind(**options).ainvoke(messages)
    except BaseException:
        await _record(node, response, reservation, uncertain=True)
        raise
    await _record(node, response, reservation)
    return response.content


async def astream_tracked(messages: list[BaseMessage], node: str) -> str:
    reservation = await _reserve(messages, settings.llm_max_output_tokens)
    full = None
    try:
        async for chunk in get_llm().bind(max_tokens=settings.llm_max_output_tokens).astream(messages):
            if chunk.content:
                emit(Token(node=node, text=chunk.content))
            full = chunk if full is None else full + chunk
    except BaseException:
        await _record(node, full, reservation, uncertain=True)
        raise
    await _record(node, full, reservation)
    return full.content if full is not None else ""
