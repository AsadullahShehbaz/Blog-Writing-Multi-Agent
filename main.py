"""
main.py
────────────────────────────────────────────────────────────
ENTRY POINT — the public interface for running the blog agent.

Usage:
    python main.py                          # runs default topic
    python main.py "Write a blog on RAG"    # custom topic

Or import and call from a notebook:
    from main import run
    result = run("Self-Attention in Transformers")
"""

from __future__ import annotations

import sys
from datetime import date
from typing import Optional

from dotenv import load_dotenv

# Load OPENAI_API_KEY and TAVILY_API_KEY from .env file
load_dotenv()

from schemas import Plan
from graph import app


# ─────────────────────────────────────────────
# INITIAL STATE FACTORY
# ─────────────────────────────────────────────
def _initial_state(topic: str, as_of: str) -> dict:
    """
    Creates the starting State dict.
    All fields must be present even if empty — LangGraph validates the schema.

    Defaults:
    - recency_days=7 is a placeholder; router_node will overwrite it.
    - sections=[] starts empty; workers append to it via operator.add.
    """
    return {
        "topic":          topic,
        "mode":           "",       # set by router
        "needs_research": False,    # set by router
        "queries":        [],       # set by router
        "evidence":       [],       # set by research (if reached)
        "plan":           None,     # set by orchestrator
        "as_of":          as_of,
        "recency_days":   7,        # placeholder; router overwrites
        "sections":       [],       # workers append here
        "final":          "",       # set by reducer
    }


# ─────────────────────────────────────────────
# MAIN RUN FUNCTION
# ─────────────────────────────────────────────
def run(topic: str, as_of: Optional[str] = None) -> dict:
    """
    Runs the full blog generation pipeline.

    Args:
        topic:  The blog topic (e.g., "Write a blog on Self-Attention")
        as_of:  Reference date string "YYYY-MM-DD" (defaults to today)

    Returns:
        The final state dict. Key fields:
          out["final"]    → complete Markdown string
          out["plan"]     → Plan object (title, tasks, etc.)
          out["evidence"] → list of EvidenceItem objects used
    """
    if as_of is None:
        as_of = date.today().isoformat()

    out = app.invoke(_initial_state(topic, as_of))

    # ── Pretty Print Summary ──────────────────────────────────────
    plan: Plan = out["plan"]
    sep = "=" * 90

    print(f"\n{sep}")
    print(f"  TOPIC         : {topic}")
    print(f"  AS_OF         : {out.get('as_of')}   RECENCY_DAYS: {out.get('recency_days')}")
    print(f"  MODE          : {out.get('mode')}")
    print(f"  BLOG_KIND     : {plan.blog_kind}")
    print(f"  NEEDS_RESEARCH: {out.get('needs_research')}")
    print(f"  QUERIES       : {(out.get('queries') or [])[:4]}")
    print(f"  EVIDENCE_COUNT: {len(out.get('evidence', []))}")
    print(f"  TASKS         : {len(plan.tasks)}")
    print(f"  OUTPUT_CHARS  : {len(out.get('final', ''))}")
    print(f"  SAVED AS      : {plan.blog_title}.md")
    print(f"{sep}\n")

    return out


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # If a topic is passed as CLI arg, use it; otherwise use default
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI for kids"
    run(topic)
