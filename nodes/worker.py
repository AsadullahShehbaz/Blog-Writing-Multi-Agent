"""
nodes/worker.py
────────────────────────────────────────────────────────────
WORKER NODE — writes ONE section of the blog post.

Multiple workers run IN PARALLEL (via LangGraph's Send API).
Each worker receives its own Task payload (not the full State).

This is the "journalist" writing their assigned story.
"""

from __future__ import annotations
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from schemas import Task, Plan, EvidenceItem
from utils import llm


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
WORKER_SYSTEM = r"""You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the Goal and cover ALL Bullets in order (do not skip or merge).
- Stay within ±15% of Target words.
- Output ONLY the section content in Markdown — no blog title, no extra commentary.
- Start with a '## <Section Title>' heading.

Scope guard (prevents drift):
- If blog_kind == "news_roundup": do NOT turn this into a tutorial.
  Focus on summarizing events and implications only.

Citation rules:
- mode == open_book: for EVERY specific event/company/model/funding claim,
  attach a Markdown link: ([Source](URL)). Only use URLs from Evidence.
  If unsupported, write: "Not found in provided sources."
- requires_citations == true (hybrid): cite Evidence URLs for outside-world claims.
- Evergreen reasoning (concepts, math) is fine without citations.

Code:
- If requires_code == true, include at least one minimal, correct code snippet.

Math and formulas:
- Write all math using dollar sign notation ONLY:
  - Inline: $Q = X W_Q$
  - Block: $$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- NEVER use a bare $ sign for anything other than math (e.g., never write $-1e9 or $_variable).
  If you need to write a dollar amount or constant, write it as plain text: "-1e9" or "1 billion".
- Every $ must have a closing $, no exceptions.
- Do not nest $$ inside $, and do not mix notations.
Style:
- Short paragraphs, bullets where helpful, fenced code blocks for code.
- Be precise and implementation-oriented. No fluff or marketing language.
"""

# ─────────────────────────────────────────────
# NODE FUNCTION
# ─────────────────────────────────────────────
def worker_node(payload: dict) -> dict:
    """
    Unlike other nodes, worker receives a PAYLOAD dict (not full State).
    This payload is constructed by the fanout() function in graph/fanout.py.

    Reads:  payload["task"], payload["plan"], payload["evidence"], etc.
    Writes: {"sections": [(task.id, section_markdown)]}

    The (id, markdown) tuple allows the reducer to sort sections correctly
    even though workers run in parallel (order is non-deterministic).
    """
    # Reconstruct typed objects from raw dicts
    task:     Task          = Task(**payload["task"])
    plan:     Plan          = Plan(**payload["plan"])
    evidence: List[EvidenceItem] = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    # Format bullets as a readable list for the prompt
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Compact evidence list for citation (max 20 items, short format)
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}"
            for e in evidence[:20]
        )

    section_md: str = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Constraints: {plan.constraints}\n"
            f"Topic: {payload['topic']}\n"
            f"Mode: {payload.get('mode', 'closed_book')}\n"
            f"As-of: {payload.get('as_of')} (recency_days={payload.get('recency_days')})\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Tags: {task.tags}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n\n"
            f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
        )),
    ]).content.strip()

    # Return a list with one tuple: (task_id, markdown_string)
    # The Annotated[..., operator.add] in State will APPEND this
    # to sections collected from other workers.
    return {"sections": [(task.id, section_md)]}
