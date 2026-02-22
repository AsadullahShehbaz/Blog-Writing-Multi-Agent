"""
graph/builder.py
────────────────────────────────────────────────────────────
GRAPH BUILDER — wires all nodes and edges into a compiled LangGraph.

A LangGraph StateGraph is like a flowchart where:
  - Nodes  = functions that read/write state
  - Edges  = connections between nodes (fixed or conditional)

The compiled `app` object is what you actually invoke with app.invoke(...)

GRAPH TOPOLOGY:
  START
    │
    ▼
  [router]  ──── needs_research=False ────► [orchestrator]
    │                                              │
    │ needs_research=True                          │
    ▼                                              │
  [research] ──────────────────────────────► [orchestrator]
                                                   │
                                          fanout() ↓ (parallel)
                                    ┌──── [worker] ────┐
                                    │     [worker]     │
                                    │     [worker]     │
                                    └──────────────────┘
                                                   │
                                                   ▼
                                              [reducer]
                                                   │
                                                  END
"""

from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from schemas import State
from nodes import (
    router_node, route_next,
    research_node,
    orchestrator_node,
    worker_node,
    reducer_node,
)
from graph.fanout import fanout


def build_graph():
    """
    Constructs and compiles the LangGraph StateGraph.

    Returns a compiled graph app ready to invoke.
    Calling this function once and reusing `app` is efficient
    (compilation happens here, not at invoke time).
    """
    g = StateGraph(State)

    # ── Register Nodes ──────────────────────────────────────────
    g.add_node("router",       router_node)
    g.add_node("research",     research_node)
    g.add_node("orchestrator", orchestrator_node)
    g.add_node("worker",       worker_node)
    g.add_node("reducer",      reducer_node)

    # ── Fixed Edges ──────────────────────────────────────────────
    g.add_edge(START,       "router")       # always start with routing
    g.add_edge("research",  "orchestrator") # after research, always plan
    g.add_edge("reducer",   END)            # reducer is always last

    # ── Conditional Edge: router → research OR orchestrator ──────
    g.add_conditional_edges(
        "router",       # source node
        route_next,     # function that returns the next node name
        {               # mapping: return value → node name
            "research":     "research",
            "orchestrator": "orchestrator",
        },
    )

    # ── Fanout Edge: orchestrator → multiple workers ─────────────
    # fanout() returns a List[Send] — each Send = one parallel worker
    g.add_conditional_edges(
        "orchestrator",
        fanout,
        ["worker"],   # list of possible target nodes (just "worker" here)
    )

    # ── After ALL workers finish → reducer ───────────────────────
    # LangGraph automatically waits for all parallel workers to
    # complete before running reducer (synchronization point).
    g.add_edge("worker", "reducer")

    return g.compile()


# Build once at module import time
app = build_graph()
