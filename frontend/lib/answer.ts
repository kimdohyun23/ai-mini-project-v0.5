import type { AgentEvent } from "./events";

export function getCurrentAnswer(events: readonly AgentEvent[]): string {
  let answer = "";
  for (const event of events) {
    if (event.type === "node_start" && event.node === "writer") {
      answer = "";
    } else if (event.type === "token" && event.node === "writer") {
      answer += event.text;
    } else if (event.type === "report") {
      answer = event.markdown;
    }
  }
  return answer;
}
