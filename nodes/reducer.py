"""
nodes/reducer.py
────────────────────────────────────────────────────────────
REDUCER NODE — the final step that assembles the blog post.

After all workers finish, their sections are collected in state["sections"]
as a list of (task_id, markdown) tuples — but in RANDOM order because
workers run in parallel.

The reducer:
  1. Sorts sections by task_id (restores outline order)
  2. Joins them into one Markdown document
  3. Saves the file to disk
"""

from __future__ import annotations
from pathlib import Path

from schemas import State
import re 

def reducer_node(state: State) -> dict:
    """
    Reads:  state["plan"], state["sections"]
    Writes: state["final"]
    Side effect: saves a .md file to disk

    Why sort? Workers run in parallel → section 3 might finish before section 1.
    Sorting by task.id restores the logical order defined by the orchestrator.
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("reducer_node called but state['plan'] is None.")

    # Sort by task_id (first element of each tuple)
    ordered_sections = [
        md
        for _, md in sorted(state["sections"], key=lambda pair: pair[0])
    ]

    # Join sections with double newline between them
    body = "\n\n".join(ordered_sections).strip()

    # Add H1 title at the top
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    # Save to disk in current working directory
    filename = plan.blog_title.lower()
    filename = filename[:40]  
    filename = re.sub(r'[\\/:*?"<>|]', '_', plan.blog_title) + ".md"
    Path(filename).write_text(final_md, encoding="utf-8")

    return {"final": final_md}
