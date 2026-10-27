"""
그래프 구성
──────────────────────────────────────────────────────────────
  START → supervisor → (researcher | competitor_analyst | writer)
  researcher, competitor_analyst → supervisor
  writer → step_evaluator → supervisor → (END | writer)
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agent.nodes import (
    competitor_analyst,
    competitor_branch,
    competitor_join,
    fan_out_competitors,
    researcher,
    step_evaluator,
    writer,
)
from app.agent.state import AgentState
from app.agent.supervisor import route_from_supervisor, supervisor


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("competitor_analyst", competitor_analyst)
    g.add_node("competitor_branch", competitor_branch)
    g.add_node("competitor_join", competitor_join)
    g.add_node("writer", writer)
    g.add_node("step_evaluator", step_evaluator)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"researcher": "researcher", "competitor_analyst": "competitor_analyst", "writer": "writer", "end": END},
    )
    g.add_edge("researcher", "supervisor")
    g.add_conditional_edges("competitor_analyst", fan_out_competitors, ["competitor_branch", "competitor_join"])
    g.add_edge("competitor_branch", "competitor_join")
    g.add_edge("competitor_join", "supervisor")
    g.add_edge("writer", "step_evaluator")
    g.add_edge("step_evaluator", "supervisor")

    return g.compile()


graph = build_graph()
