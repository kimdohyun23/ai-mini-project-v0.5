# DAY 2 실행과 검증

과제 출처: https://actually-war-1ea.notion.site/Supervisor-4047f4c8669380a0b6d4ff8abbf2ed53

## 작업 폴더

현재 저장소의 day2 브랜치에서 실행합니다. 옆의 ai-mini-baseline-ref는 main 원본 비교용이며 실행하거나 수정하지 않습니다.

## 구현 범위

- Supervisor: LLM이 질문 의도와 검색/분석 상태를 보고 다음 작업을 선택합니다. 검색은 최대 2회입니다. writer의 평가 결과도 Supervisor로 돌아옵니다.
- 기업 분석: LangGraph Send로 기업별 갈래를 동시 실행합니다. 기업별 결과를 reducer로 병합하고 실패한 기업도 상태로 기록합니다. 갈래별 제한 시간은 기본 45초입니다.
- LLM 비용: 실행별 ContextVar와 공유 예약 장부를 사용합니다. 병렬 호출은 예약 중인 비용도 포함하여 승인합니다. 입력 UTF-8 바이트 수와 출력 토큰 상한을 보수적으로 예약하고 사용량 메타데이터로 정산합니다.
- 예산 부족: 추가 LLM 호출을 시작하지 않고 수집한 결과를 부분 보고서로 보존합니다. 완료된 보고서로 표시하지 않습니다.
- 사용량을 알 수 없는 실패/취소는 예약 상한으로 보수적으로 기록하고 추가 호출을 막습니다. 이것은 확정 청구 금액이 아닙니다.

## 설정

기존 .env 키는 그대로 사용합니다. 아래 선택 설정은 환경변수 또는 .env로 전달할 수 있습니다.

```text
LLM_COST_LIMIT_USD=0.01
LLM_MAX_OUTPUT_TOKENS=1600
COMPETITOR_TIMEOUT_SECONDS=45
```

Supervisor 출력은 최대 350토큰, 기업별 분석은 최대 650토큰, writer는 LLM_MAX_OUTPUT_TOKENS를 적용합니다. 자동 재시도는 비활성화합니다.
비용 범위는 config.py 단가표에 따른 텍스트 LLM 예상 비용이며, 임베딩과 Tavily 요금은 포함하지 않습니다. 단가표에 없는 모델은 호출을 거절합니다.

기본 상한 $0.01은 DAY 1 Q1/Q2/Q3 화면 비용 약 $0.00185/$0.00190/$0.00193에 추가 Supervisor 호출과 보수적인 예약 비용을 고려한 설정입니다. 정확한 청구서 비용을 보증하는 값은 아닙니다.

## 실행

프로젝트 루트에서 서로 다른 두 터미널을 사용합니다.

Backend:
```bash
cd backend
uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Frontend:
```bash
cd frontend
NEXT_TELEMETRY_DISABLED=1 npm run dev -- --hostname 127.0.0.1
```

화면은 http://localhost:3000 입니다. 두 서버를 중복 실행하지 않습니다.

## 자동 검증

프로젝트 루트에서:
```bash
PYTHONPATH=backend uv run --locked python -m unittest discover -s tests -p 'test_day2_*.py' -v
uv run --locked ruff check --no-cache backend/app/agent backend/app/budget.py backend/app/config.py backend/app/llm.py backend/app/main.py tests
```

테스트는 외부 API 호출을 대체하여 라우팅, 재검색 상한, 세 갈래의 동시 실행, 실패/시간 초과, 중복 병합, 비용 예약/정산, 요청별 격리, 낮은 예산의 SSE 응답을 확인합니다.

## 낮은 상한의 수동 검증

일반 8000 서버와 별도로 backend 폴더에서 실행할 수 있습니다.
```bash
LLM_COST_LIMIT_USD=0.000001 uv run --locked uvicorn app.main:app --host 127.0.0.1 --port 8001
```

다른 터미널에서:
```bash
curl -N -H 'Content-Type: application/json' -d '{"item_id":"hbf","query":"HBF는 HBM과 무엇이 다른가요?","test_mode":false}' http://127.0.0.1:8001/api/chat
```

기대 결과는 호출 전 중단 로그, LLM 호출 0회, 비용 $0, 부분 결과임을 명시한 보고서입니다. 이 명령은 .env를 변경하지 않습니다. 종료는 해당 테스트 서버 터미널에서 Control+C입니다.

## 제출

파일명: ai-mini-DAY2-판교7반-김도현.zip
내용: day2.diff, evidence/, results.md

필수 증거: Q1/Q2/Q3 Frontend, Q2 기업별 병렬 갈래 추적, 낮은 비용 상한의 로그 또는 화면.
캡처는 사용자가 직접 실행한 원본을 보존합니다. 검증용 API 실행 로그와 캡처 실행의 run_id/측정값을 섞지 않습니다.
수정을 모두 커밋한 다음 `git diff main > day2.diff`로 변경 내역을 만듭니다. 비밀키, 벡터 인덱스, DAY 1 캡처와 Pages 문서를 스테이징하지 않습니다.

평가 기준 자체는 Baseline의 길이 검사 상태입니다. 근거성과 내용 품질 평가 개선은 이번 DAY 2 범위에서 제외합니다.
