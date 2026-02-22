"""
nodes/orchestrator.py
────────────────────────────────────────────────────────────
ORCHESTRATOR NODE — the "brain" that creates the blog outline.

Receives: topic, mode, evidence (may be empty)
Produces: a Plan with 5–9 Tasks (sections)

Think of this as a senior editor assigning stories to journalists.
Each Task = one story assignment (title + goal + bullets).
Workers will write those stories in parallel.
"""

from __future__ import annotations

from langchain_core.messages import SystemMessage, HumanMessage

from schemas import State, Plan
from utils import llm


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks).
- Each task must have: goal (1 sentence), 3–6 concrete bullets, target word count (120–550).

Quality bar (include at least 2 of these across sections):
  * Minimal code sketch / MWE  →  set requires_code=True
  * Edge cases / failure modes
  * Performance or cost considerations
  * Security/privacy considerations (if relevant)
  * Debugging/observability tips

Mode rules:
- closed_book: keep it evergreen; ignore evidence.
- hybrid: use evidence for up-to-date tool/model names in bullets;
          mark those sections requires_research=True, requires_citations=True.
- open_book: blog_kind MUST be "news_roundup".
             Every section summarizes events + implications.
             DO NOT include tutorial/how-to sections.
             If evidence is empty, say so transparently.

Output must match the Plan schema exactly.
"""


# ─────────────────────────────────────────────
# NODE FUNCTION
# ─────────────────────────────────────────────
def orchestrator_node(state: State) -> dict:
    """
    Reads:  topic, mode, as_of, recency_days, evidence
    Writes: plan

    The plan becomes the "task list" that fanout() sends to workers.
    """
    planner = llm.with_structured_output(Plan)
    mode     = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    # Tell the LLM to force news_roundup kind for open_book mode
    force_news_note = "Force blog_kind=news_roundup" if mode == "open_book" else ""

    plan: Plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n"
            f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
            f"{force_news_note}\n\n"
            f"Evidence (use only for fresh claims; may be empty):\n"
            f"{[e.model_dump() for e in evidence][:16]}\n\n"
            f"Instruction: If mode=open_book, do NOT write a tutorial."
        )),
    ])

    # Safety net: ensure open_book always sets news_roundup kind
    if mode == "open_book":
        plan.blog_kind = "news_roundup"

    return {"plan": plan}
