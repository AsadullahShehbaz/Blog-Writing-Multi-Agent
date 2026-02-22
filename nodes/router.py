"""
nodes/router.py
────────────────────────────────────────────────────────────
ROUTER NODE — the very first step in the graph.

Its job: look at the topic and decide HOW to handle it.
  - Does it need live web data?
  - How old can the sources be?
  - What search queries should we run?

This avoids wasting API calls on Tavily for topics
that don't need real-time info (e.g., "explain backprop").
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from schemas import State, RouterDecision
from utils import llm


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics — correctness does NOT depend on recent facts (concepts, math, algorithms).

- hybrid (needs_research=true):
  Mostly evergreen but benefits from up-to-date examples/tools/model names.

- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week in AI", "latest releases", pricing, policy.

If needs_research=true:
- Output 3–10 high-signal queries.
- Be specific (avoid single-word queries like "AI" or "LLM").
- For open_book weekly roundups, scope queries to the last 7 days.
"""


# ─────────────────────────────────────────────
# NODE FUNCTION
# ─────────────────────────────────────────────
def router_node(state: State) -> dict:
    """
    Reads:  state["topic"], state["as_of"]
    Writes: mode, needs_research, queries, recency_days

    Uses `with_structured_output(RouterDecision)` — this tells the LLM
    to return a JSON object that matches the RouterDecision Pydantic schema.
    No manual JSON parsing needed!
    """
    decider = llm.with_structured_output(RouterDecision)

    decision: RouterDecision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
    ])

    # Set the recency window based on the mode chosen
    # open_book  → only last 7 days are fresh enough
    # hybrid     → last 45 days is fine
    # closed_book→ 10 years (effectively unlimited)
    recency_map = {
        "open_book":   7,
        "hybrid":      45,
        "closed_book": 3650,
    }
    recency_days = recency_map.get(decision.mode, 3650)

    return {
        "needs_research": decision.needs_research,
        "mode":           decision.mode,
        "queries":        decision.queries,
        "recency_days":   recency_days,
    }


# ─────────────────────────────────────────────
# CONDITIONAL EDGE FUNCTION
# ─────────────────────────────────────────────
def route_next(state: State) -> str:
    """
    Called by the graph after router_node.
    Returns the name of the NEXT node to visit.

    This is a "conditional edge" — the route depends on runtime state.
    If needs_research=True  → go to "research" node
    If needs_research=False → skip straight to "orchestrator"
    """
    return "research" if state["needs_research"] else "orchestrator"
