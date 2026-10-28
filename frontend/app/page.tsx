"use client";

/**
 * 기술 전략 분석 에이전트 화면
 * ──────────────────────────────────────────────
 * 왼쪽 : 질문 입력, 답변 스트리밍, 최종 보고서
 * 오른쪽 : 실행 관찰 패널 (노드 타임라인, 경쟁사별 갈래, 라우팅, 평가, LLM 사용량, 진행 로그)
 *
 * Backend가 보내는 이벤트(lib/events.ts)를 모두 events 배열에 쌓아두고,
 * 화면의 각 패널은 그 배열에서 필요한 이벤트만 골라 보여줍니다.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { AgentEvent, ItemConfig } from "@/lib/events";
import { API_BASE, streamChat } from "@/lib/sse";
import { getCurrentAnswer } from "@/lib/answer";

type TimedRow = { name: string; start: number; duration?: number };

/** *_start / *_end 이벤트를 짝지어 "시작 시점 + 소요 시간" 행으로 만듭니다. */
function pairRows(events: AgentEvent[], startType: string, endType: string, key: (e: any) => string, t0: number): TimedRow[] {
  const rows: TimedRow[] = [];
  const open: Record<string, number[]> = {};
  for (const e of events as any[]) {
    if (e.type === startType) {
      rows.push({ name: key(e), start: e.ts - t0 });
      (open[key(e)] ??= []).push(rows.length - 1);
    } else if (e.type === endType) {
      const idx = open[key(e)]?.shift();
      if (idx !== undefined) rows[idx].duration = e.duration_ms;
    }
  }
  return rows;
}

const ms = (v?: number) => (v === undefined ? "실행 중" : `${v.toLocaleString()} ms`);

export default function Home() {
  const [config, setConfig] = useState<Record<string, ItemConfig>>({});
  const [itemId, setItemId] = useState("");
  const [query, setQuery] = useState("");
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [running, setRunning] = useState(false);
  const [testMode, setTestMode] = useState(false);   // 점검용 테스트 모드 (부실한 결과 생성)
  const [loadError, setLoadError] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  // 아이템 설정 불러오기
  useEffect(() => {
    fetch(`${API_BASE}/api/config`)
      .then((r) => r.json())
      .then((c: Record<string, ItemConfig>) => {
        setConfig(c);
        setItemId(Object.keys(c)[0] ?? "");
      })
      .catch(() => setLoadError("Backend에 연결할 수 없습니다. 서버 실행 여부와 NEXT_PUBLIC_API_BASE 를 확인하세요."));
  }, []);

  const item = config[itemId];

  async function run() {
    if (!itemId || !query.trim()) return;
    setEvents([]);
    setRunning(true);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    try {
      await streamChat({ item_id: itemId, query, test_mode: testMode }, (ev) => setEvents((prev) => [...prev, ev]), ctrl.signal);
    } catch (err) {
      if (!ctrl.signal.aborted) {
        setEvents((prev) => [...prev, { type: "error", ts: Date.now(), message: String(err) }]);
      }
    } finally {
      setRunning(false);
    }
  }

  // ── 이벤트 → 화면 데이터 ─────────────────────────
  const view = useMemo(() => {
    const t0 = events.find((e) => e.type === "run_start")?.ts ?? 0;
    const pick = <T extends AgentEvent["type"]>(t: T) =>
      events.filter((e): e is Extract<AgentEvent, { type: T }> => e.type === t);
    const usage = pick("llm_usage");
    return {
      t0,
      runStart: pick("run_start").at(-1),
      nodes: pairRows(events, "node_start", "node_end", (e) => e.node, t0),
      branches: pairRows(events, "branch_start", "branch_end", (e) => e.branch, t0),
      routes: pick("route"),
      evaluations: pick("evaluation"),
      progress: pick("progress"),
      errors: pick("error"),
      usage,
      usageTotal: usage.reduce(
        (a, u) => ({ input: a.input + u.input_tokens, output: a.output + u.output_tokens, cost: a.cost + u.cost_usd }),
        { input: 0, output: 0, cost: 0 },
      ),
      answer: getCurrentAnswer(events),
      report: pick("report").at(-1),
      done: pick("done").at(-1),
    };
  }, [events]);

  return (
    <main className="layout">
      {/* ───────── 왼쪽: 입력과 결과 ───────── */}
      <section className="column">
        <h1>기술 전략 분석 에이전트</h1>
        {loadError && <p className="error">{loadError}</p>}

        <div className="card">
          <label>
            분석 아이템
            <select value={itemId} onChange={(e) => setItemId(e.target.value)} disabled={running}>
              {Object.entries(config).map(([id, c]) => (
                <option key={id} value={id}>{c.name}</option>
              ))}
            </select>
          </label>
          {item && <p className="muted">분석 대상 기업: {item.competitors.join(", ")}</p>}

          {item && (
            <div className="questions">
              {Object.entries(item.test_questions).map(([k, q]) => (
                <button key={k} className="ghost" onClick={() => setQuery(q)} disabled={running}>
                  {k.toUpperCase()} 테스트 질문
                </button>
              ))}
            </div>
          )}

          <div className="mode">
            <span className="muted">실행 모드</span>
            <button className={testMode ? "ghost" : ""} onClick={() => setTestMode(false)} disabled={running}>정상 모드</button>
            <button className={testMode ? "warn" : "ghost"} onClick={() => setTestMode(true)} disabled={running}>테스트 모드</button>
          </div>
          {testMode && <p className="muted">테스트 모드: 근거 출처와 기업별 TRL/인용을 누락하고 부실 보고서를 생성합니다. 평가와 재시도 상한 점검용입니다.</p>}

          <textarea rows={3} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="질문을 입력하세요" disabled={running} />
          <div className="actions">
            <button onClick={run} disabled={running || !query.trim()}>{running ? "실행 중..." : "실행"}</button>
            {running && <button className="ghost" onClick={() => abortRef.current?.abort()}>중지</button>}
          </div>
        </div>

        <div className="card">
          <h2>답변 (스트리밍)</h2>
          <pre className="answer">{view.answer || (running ? "작성 대기 중..." : "")}</pre>
        </div>

        {view.report && (
          <div className="card">
            <h2>최종 보고서</h2>
            <p className="muted">run_id: {view.report.run_id}</p>
            <pre className="answer">{view.report.markdown}</pre>
          </div>
        )}
      </section>

      {/* ───────── 오른쪽: 실행 관찰 패널 ───────── */}
      <section className="column">
        {view.runStart && (
          <div className={`card mode-banner ${view.runStart.test_mode ? "test" : "normal"}`}>
            실행 모드 : <b>{view.runStart.test_mode ? "테스트 모드 (부실한 결과 생성)" : "정상 모드"}</b>
            <span className="muted"> · run_id {view.runStart.run_id}</span>
          </div>
        )}
        {view.errors.map((e, i) => <p key={i} className="error">오류: {e.message}</p>)}

        {view.done && (
          <div className="card summary">
            <div><span>전체 소요</span><b>{ms(view.done.total_duration_ms)}</b></div>
            <div><span>LLM 호출</span><b>{view.done.llm_calls}회</b></div>
            <div><span>토큰 (입력/출력)</span><b>{view.done.input_tokens.toLocaleString()} / {view.done.output_tokens.toLocaleString()}</b></div>
            <div><span>예상 비용</span><b>${view.done.total_cost_usd.toFixed(4)}</b></div>
          </div>
        )}

        <div className="card">
          <h2>노드 실행 타임라인</h2>
          <table>
            <thead><tr><th>노드</th><th>시작 (실행 후)</th><th>소요</th></tr></thead>
            <tbody>
              {view.nodes.map((r, i) => (
                <tr key={i}><td>{r.name}</td><td>{ms(r.start)}</td><td>{ms(r.duration)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>경쟁사별 분석 갈래</h2>
          <table>
            <thead><tr><th>기업</th><th>시작 (실행 후)</th><th>소요</th></tr></thead>
            <tbody>
              {view.branches.map((r, i) => (
                <tr key={i}><td>{r.name}</td><td>{ms(r.start)}</td><td>{ms(r.duration)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>라우팅</h2>
          <ol className="list">
            {view.routes.map((r, i) => <li key={i}>{r.source} → <b>{r.target}</b> <span className="muted">({r.reason})</span></li>)}
          </ol>
        </div>

        <div className="card">
          <h2>단계별 평가</h2>
          <ol className="list">
            {view.evaluations.map((e, i) => (
              <li key={i}>[{e.stage} · {e.attempt}회차] <b className={e.passed ? "pass" : "fail"}>{e.passed ? "통과" : "미통과"}</b> {e.reason}</li>
            ))}
          </ol>
        </div>

        <div className="card">
          <h2>LLM 사용량</h2>
          <table>
            <thead><tr><th>노드</th><th>입력</th><th>출력</th><th>비용(USD)</th></tr></thead>
            <tbody>
              {view.usage.map((u, i) => (
                <tr key={i}><td>{u.node}</td><td>{u.input_tokens}</td><td>{u.output_tokens}</td><td>{u.cost_usd.toFixed(5)}</td></tr>
              ))}
              {view.usage.length > 0 && (
                <tr className="total"><td>합계</td><td>{view.usageTotal.input}</td><td>{view.usageTotal.output}</td><td>{view.usageTotal.cost.toFixed(5)}</td></tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="card">
          <h2>진행 로그</h2>
          <ol className="list log">
            {view.progress.map((p, i) => (
              <li key={i}><span className="muted">+{(p.ts - view.t0).toLocaleString()} ms</span> [{p.node}] {p.message}</li>
            ))}
          </ol>
        </div>
      </section>
    </main>
  );
}
