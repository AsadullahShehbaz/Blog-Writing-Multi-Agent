"""
nodes/research.py
────────────────────────────────────────────────────────────
RESEARCH NODE — runs web searches and filters/deduplicates results.

Only reached when router_node sets needs_research=True.

Two-step process:
  1. Run all queries through Tavily (raw results)
  2. Ask LLM to deduplicate & normalize into EvidenceItem objects
  3. For open_book mode: apply a HARD DATE FILTER (drop stale results)
"""

from __future__ import annotations
from datetime import date, timedelta
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage

from schemas import State, EvidenceItem, EvidencePack
from utils import llm, tavily_search, iso_to_date


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer authoritative sources: company blogs, official docs, reputable outlets.
- Normalize published_at to ISO format (YYYY-MM-DD) if inferrable from title/snippet.
  If you cannot infer a reliable date, set published_at=null. Do NOT guess.
- Keep snippets concise (< 2 sentences).
- Deduplicate by URL.
"""


# ─────────────────────────────────────────────
# NODE FUNCTION
# ─────────────────────────────────────────────
def research_node(state: State) -> dict:
    """
    Reads:  state["queries"], state["mode"], state["as_of"], state["recency_days"]
    Writes: state["evidence"]

    Steps:
    ① Collect raw results from Tavily for each query
    ② Use LLM to extract + deduplicate into EvidenceItem list
    ③ Apply hard date filter for open_book mode
    """

    # ① Gather raw search results (max 10 queries, 6 results each)
    queries = (state.get("queries") or [])[:2]
    raw_results: List[dict] = []
    for query in queries:
        raw_results.extend(tavily_search(query, max_results=2))

    # Nothing found → return empty evidence (graph still continues)
    if not raw_results:
        return {"evidence": []}

    # ② Ask LLM to normalize & deduplicate
    extractor = llm.with_structured_output(EvidencePack)
    pack: EvidencePack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=(
            f"As-of date: {state['as_of']}\n"
            f"Recency days: {state['recency_days']}\n\n"
            f"Raw results:\n{raw_results}"
        )),
    ])

    # Deduplicate by URL using a dict (last write wins)
    deduped: dict[str, EvidenceItem] = {
        e.url: e for e in pack.evidence if e.url
    }
    evidence: List[EvidenceItem] = list(deduped.values())

    # ③ Hard date filter — ONLY for open_book (weekly news roundup)
    # For hybrid/closed_book, we keep all results even without dates.
    if state.get("mode") == "open_book":
        as_of_date  = date.fromisoformat(state["as_of"])
        cutoff_date = as_of_date - timedelta(days=int(state["recency_days"]))

        fresh_evidence: List[EvidenceItem] = []
        for item in evidence:
            item_date = iso_to_date(item.published_at)
            # Keep only items with a parseable date within the window
            if item_date and item_date >= cutoff_date:
                fresh_evidence.append(item)

        evidence = fresh_evidence

    return {"evidence": evidence}
