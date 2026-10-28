# DAY 3: Evaluation 개선

과제: https://actually-war-1ea.notion.site/Evaluation-4047f4c8669380e99f93cf611290d538

## 기준점과 범위

Day 2 최종 커밋 `e7bd0fe3863c1337c161c42cefc97f7502723b3d`에서 `day3` 브랜치를 만들었다. 의존성과 .env는 변경하지 않는다. 비용 상한은 실행당 $0.01을 유지한다. 평가 Judge도 Day 2와 같은 비용 예약 및 정산 함수를 사용한다.

## 평가 책임과 흐름

```text
Supervisor -> researcher -> research_evaluator -> Supervisor
Supervisor -> competitor_analyst -> 기업별 병렬 갈래 -> competitor_join
           -> analysis_evaluator -> Supervisor
Supervisor -> writer -> step_evaluator(최종 5항목) -> Supervisor -> 종료/재작성
```

생성자가 자기 결과를 통과시키지 않도록 별도 평가 노드를 둔다. Supervisor는 평가 결과와 횟수에 따라 재시도를 지시한다. 중간 단계에는 추가 LLM 호출 없이 검사할 수 있는 조건을 사용한다. 개수와 형식 검사는 내용의 진실성을 보증하지 않으므로 최종 의미 평가와 구분한다.

- 근거 수집: 서로 다른 근거 3건 이상, 모든 근거의 출처와 본문 존재.
- 기업 분석: 기대 기업 집합과 결과 집합 동일, 각 기업에 TRL 1~9와 유효 번호 또는 판단 불가와 자료 부족 사유, 실패 갈래 없음.
- 첫 시도 포함 단계별 최대 2회: 재시도는 최대 1회. 미통과 이유는 검색 질의 또는 분석 프롬프트에 포함하고 진행 로그에 남긴다.
- 중간 단계 상한 이후: 미통과 상태와 사유를 보존하고 제한된 자료로 다음 평가까지 진행한다. 미검증 결과를 통과로 바꾸지 않는다. 최종 미통과 초안은 검토 필요로 표시한다.
- 테스트 모드: 검색 결과 출처 제거, 기업 분석의 TRL/인용 누락, 기존 고정 부실 보고서 사용. 고의 실패 주입임을 진행 로그에 표시한다. 실제 공급자 장애라고 주장하지 않는다.

근거 3건은 단일 자료에만 의존하는 상황을 탐지하기 위한 최소 검사값이다. 정상 Q2 실제 검색 결과 수와 대조하며, 3건이라는 수가 검색 관련성이나 출처 다양성을 보증하지는 않는다.

## 인용 번호 전파

각 기업은 자기 검색 목록의 로컬 [n]으로 분석한다. join에서 실제 사용한 근거를 전체 근거 목록에 합치고, `(출처, 본문)`으로 중복을 구분하여 로컬 번호를 전역 번호로 한 번만 변환한다. 같은 PDF의 다른 문단은 별개 근거다. 잘못된 로컬 번호는 [0]으로 남겨 검사에서 걸리게 한다. 재시도에서도 기존 전역 번호는 유지된다.

Writer와 Judge는 번호가 같은 근거 목록의 동일한 본문 앞 1,200자를 받는다. Writer가 작성한 사실 문장이나 인용을 코드가 임의로 고치지는 않는다. 보고서의 근거 목록만 실제 인용 번호와 원본 출처로 결정적으로 연결한다.

## 최종 평가

| 항목 | 방법 | 기준 |
|---|---|---|
| Answer Relevance | LLM Judge | 질문의 핵심 요구에 직접 답함 |
| Faithfulness | 주장 단위 Judge + 코드 집계 | 뒷받침 주장 수 / 전체 주장 수 >= 0.80 |
| Citation Accuracy | 번호/출처 코드 검사 + 주장 단위 Judge | 번호 누락/범위 밖/출처 누락 없음, 인용된 근거가 해당 주장을 뒷받침 |
| Completeness | 코드 | 질문이 기업 비교이면 모든 기업별 제목에 TRL과 근거 또는 판단 불가 사유 존재 |
| Uncertainty Disclosure | 한계 섹션 코드 검사 + Judge | 실질적 한계 공개, 근거 없는 TRL 단정 없음 |

Faithfulness 0.80은 과제 최소 0.70보다 보수적으로 정한 값이다. 통과를 위해 실행 결과에 맞춰 낮추지 않는다. Citation Accuracy는 인용 번호가 없는 사실 주장도 분모에 포함한다. '인용된 일부만 100%'를 전체 정확도로 표시하지 않는다.

제목과 참고 자료 목록을 제외하고 코드가 문장/한 줄의 주장 단위를 정한다. 복합 주장은 모든 부분이 지지되어야 한다. 동일한 주장 ID들을 Judge에 전달하고 누락/중복/추가 ID는 거절한다. Judge는 `supported`, `cited_support`, 고정된 원문 발췌 ID와 이유를 JSON으로 반환한다. 코드가 발췌 ID의 존재와 인용된 출처의 연결을 검사한다. 다른 자료가 주장을 뒷받침해도 인용된 자료가 아니면 Citation Accuracy 통과로 처리하지 않는다.

판정과 이유는 `final:항목명` 5개와 `report` 합계로 기존 Evaluation 이벤트에 남긴다. 최대 40개 주장까지 처리하며, 초과/잘못된 Judge 응답/시간 초과를 의미 평가 통과로 바꾸지 않는다. 상세 주장별 결과는 실행 상태의 `judge_audit`에 보존한다. 재작성 입력에는 실패 이유와 인용 누락 문장의 예를 전달한다.

## 검증 명령

프로젝트 루트에서 실행한다.

```bash
PYTHONPATH=backend uv run --locked python -m unittest discover -s tests -v
uv run --locked ruff check --no-cache backend/app/agent backend/app/main.py tests scripts/evaluate_saved_report.py
```

동일 보고서 안정성은 저장된 실제 그래프 상태 JSON으로 독립적인 Judge 호출 두 번을 비교한다. 결과를 복사하거나 캐시하지 않는다. 예:

```bash
uv run --locked python scripts/evaluate_saved_report.py --state /절대경로/state.json --output /절대경로/stability.json
```

정상/테스트 Q2, 낮은 예산, 인용 번호가 유효하지만 다른 자료를 가리키는 경우, 출처 누락, 없는 발췌 ID, 피드백 전달, 상한 종료, 병렬 결과 보존을 확인한다. 실제 API 실행은 단위 테스트와 구분한다.

## 수동 캡처와 제출

모두 HBF Q2 기준이다. 캡처 원본을 수정하거나 재현 이미지로 대체하지 않는다.

1. `1-normal-trace.png`: 정상 모드 LangSmith. research_evaluator와 analysis_evaluator, 최종 step_evaluator 흐름 확인.
2. `1-normal-eval.png`: 정상 모드 중간 평가 2곳 모두 포함.
3. `1-test-eval.png`: 테스트 모드 중간 미통과, 재시도, 상한 및 진행 로그.
4. `2-normal-eval.png`: 정상 모드 최종 5항목과 이유.
5. `2-test-eval.png`: 테스트 모드 최종 5항목. Citation Accuracy와 Completeness 미통과 확인.

모든 코드를 커밋한 뒤 `git diff day2 HEAD > day3.diff`를 프로젝트 루트에서 생성한다. 제출 ZIP은 `day3.diff`, `evidence/` 위 5장, `report_eval.md`다. 원본 API 키, .env, 문서 인덱스, 테스트 임시 파일을 넣지 않는다. 실제 측정값과 캡처 run_id를 대조한 뒤 보고서를 작성한다.

## 한계

출처가 있다는 것과 출처가 신뢰할 만하다는 것은 다르다. 웹 검색 결과 및 PDF 발췌만 평가하므로 원문의 다른 부분을 놓칠 수 있다. LLM Judge의 의미 판단도 틀릴 수 있고 원문 발췌 검사는 의미 정합성을 증명하지 않는다. 주장 분리는 문장/줄 기반이므로 완전한 원자적 주장 분할이 아니다. 정상 보고서라도 미통과일 수 있으며, 평가를 통과시키기 위해 번호를 자동으로 붙이거나 근거를 만들어내지 않는다.

## 재검토 후 수정

- 빈 초안도 작성 결과로 인식해 최초 시도 포함 2회에서 종료한다.
- TRL의 소수, 지수 표기, 분수와 범위를 정수 단계로 오인하지 않는다.
- 판단 불가의 사유는 자료 부족이나 미공개 등 설명을 요구하며 빈 항목 이름만으로 통과시키지 않는다.
- TRL 뒤의 줄바꿈과 Markdown 목록 기호는 숫자 범위로 해석하지 않는다. Judge 스키마 오류는 패널에 요약하고 상세 경로는 judge_audit에 보존한다.
- Judge 출력 예약은 주장 수에 맞춰 `min(2200, 400 + 100 * 주장 수)`로 정한다. 실제 정상 Q2의 Judge 출력 847토큰에 비해 고정 2,200토큰 예약이 과했던 점을 반영했다. 실행당 비용 상한 $0.01은 유지한다.
- 검색 결과 병합은 merge_evidence를 공통 사용한다. 기업 재시도 피드백은 분석 시작 시 한 번 기록하고 각 기업의 입력에는 계속 전달한다.
- Frontend는 Writer가 다시 시작하면 이전 초안을 지우고, 마지막 Report 이벤트의 상태와 근거 목록을 반영한다.

Frontend의 이벤트 재생 검증은 설치된 TypeScript와 Node 기본 테스트 도구만 사용한다.

```bash
frontend/node_modules/.bin/tsc tests/frontend_answer.test.ts --outDir /tmp/ai-mini-day3-tests --rootDir . --module commonjs --target ES2020 --moduleResolution node --types node --typeRoots frontend/node_modules/@types --esModuleInterop --strict --skipLibCheck
node --test /tmp/ai-mini-day3-tests/tests/frontend_answer.test.js
frontend/node_modules/.bin/tsc --noEmit --incremental false --project frontend/tsconfig.json
```

기존 캡처는 이전 커밋의 실행 기록으로 백업한다. 제출용 캡처와 보고서의 실행 정보는 최종 커밋에서 다시 수집한다.

## 인용과 Judge 근거 선택 보완

- 인용 파서를 공통 모듈 citations.py로 모아 `[1, 2]`와 `[1][2]`를 같은 방식으로 검사하고 전역 번호 및 출처 목록에 반영한다.
- 원문을 LLM이 다시 복사하게 하는 방식 대신, 생성자에게 전달한 동일 본문 앞 1,200자에 고정 발췌 ID를 부여한다. 공백에서 나누고 원문 전체를 보존한다. 여러 발췌를 함께 선택할 수 있다.
- Judge는 의미 지지 여부와 실제 발췌 ID를 반환한다. 코드가 존재하지 않는 ID, 빈 선택, 인용 출처와 다른 선택을 미통과로 처리한다. 선택 ID와 실제 본문은 judge_audit에 함께 남긴다.
- 여러 인용 중 일부 자료만으로도 주장이 입증되면 번호 집합의 완전 일치를 강요하지 않는다. 범위 밖 번호와 출처 누락은 여전히 미통과다.
- 발췌 ID가 존재한다는 것은 의미가 맞다는 보장이 아니다. 의미 판단 오류는 알려진 정답 사례와 오답 사례의 실제 Judge 실행으로 별도 점검한다. 임계값 0.80과 비용 상한 $0.01은 유지한다.

- 실제 Q2에서 발생한 주장 ID 누락/재분할을 막기 위해 Judge 응답 스키마의 필수 키를 입력 주장 ID로 고정한다. 응답 스키마의 입력 비용도 호출 전 예약에 포함한다. 의미 평가 결과는 형식 보장과 구분한다.
