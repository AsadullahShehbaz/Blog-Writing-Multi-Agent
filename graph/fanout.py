"""
graph/fanout.py
────────────────────────────────────────────────────────────
FANOUT — converts the Plan into parallel worker invocations.

This is the "map" step in a Map-Reduce pattern:
  orchestrator → fanout → [worker, worker, worker, ...] → reducer

LangGraph's `Send` API lets us dynamically create as many parallel
branches as there are tasks — without hard-coding the number of workers.

Each Send(node_name, payload) tells LangGraph:
  "Start a new instance of 'worker' node with this payload."
"""

from __future__ import annotations
from typing import List

from langgraph.types import Send

from schemas import State


def fanout(state: State) -> List[Send]:
    """
    Called as a conditional edge after orchestrator_node.

    For each Task in the Plan, creates a Send object.
    LangGraph will run all workers IN PARALLEL.

    The payload is a plain dict (not the full State) because
    each worker only needs its own task + shared context.
    This also avoids sending large state objects over the wire.
    """
    assert state["plan"] is not None, "fanout() called before plan was created."

    return [
        Send(
            "worker",       # name of the target node
            {               # payload sent to that worker instance
                "task":         task.model_dump(),          # this worker's assignment
                "topic":        state["topic"],             # original user topic
                "mode":         state["mode"],              # closed_book / hybrid / open_book
                "as_of":        state["as_of"],             # reference date
                "recency_days": state["recency_days"],      # staleness threshold
                "plan":         state["plan"].model_dump(), # full plan (for blog metadata)
                "evidence":     [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]
