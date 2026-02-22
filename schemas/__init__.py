"""
schemas/__init__.py
All Pydantic models (data shapes) used across the pipeline.
Think of these as "blueprints" for every piece of data that flows through the graph.
"""

from __future__ import annotations
from typing import List, Optional, Literal, Annotated
import operator

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─────────────────────────────────────────────
# TASK  →  one section of the blog post
# ─────────────────────────────────────────────
class Task(BaseModel):
    """
    Describes ONE section (e.g., 'Introduction', 'How Attention Works').
    The orchestrator creates a list of Tasks; each worker writes one Task.
    """
    id: int                          # Position in the outline (for ordering later)
    title: str                       # Section heading
    goal: str = Field(               # One-sentence learning objective
        ...,
        description="What the reader will understand after this section.",
    )
    bullets: List[str] = Field(      # 3–6 specific sub-points to cover
        ...,
        min_length=3,
        max_length=6,
        description="Concrete, non-overlapping subpoints.",
    )
    target_words: int = Field(..., description="Target word count (120–550).")
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False  # Should worker use evidence?
    requires_citations: bool = False # Should worker add source links?
    requires_code: bool = False      # Should worker include a code snippet?


# ─────────────────────────────────────────────
# PLAN  →  the full outline of the blog post
# ─────────────────────────────────────────────
class Plan(BaseModel):
    """
    The orchestrator node produces one Plan object.
    It contains metadata + a list of Tasks for the workers.
    """
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


# ─────────────────────────────────────────────
# EVIDENCE  →  a single web search result
# ─────────────────────────────────────────────
class EvidenceItem(BaseModel):
    """One item returned from Tavily web search."""
    title: str
    url: str
    published_at: Optional[str] = None   # ISO "YYYY-MM-DD"
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    """Container for multiple EvidenceItems (LLM outputs this)."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ─────────────────────────────────────────────
# ROUTER DECISION  →  should we search the web?
# ─────────────────────────────────────────────
class RouterDecision(BaseModel):
    """
    Router node outputs this.
    Tells the graph whether to go to research or skip straight to planning.
    """
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5, description="Results per query (3–8).")


# ─────────────────────────────────────────────
# STATE  →  the shared memory of the whole graph
# ─────────────────────────────────────────────
class State(TypedDict):
    """
    Every node reads from and writes to this shared dictionary.
    Think of it like a baton passed between runners in a relay race.

    The `sections` field uses `operator.add` — meaning each worker APPENDS
    its result instead of overwriting (parallel-safe merge).
    """
    topic: str

    # Set by router_node
    mode: str
    needs_research: bool
    queries: List[str]

    # Set by research_node
    evidence: List[EvidenceItem]

    # Set by orchestrator_node
    plan: Optional[Plan]

    # Recency control (router sets these)
    as_of: str          # ISO date, e.g. "2026-02-19"
    recency_days: int   # 7 = weekly news, 45 = hybrid, 3650 = evergreen

    # Collected by all workers in parallel — operator.add merges lists
    sections: Annotated[List[tuple[int, str]], operator.add]

    # Set by reducer_node
    final: str