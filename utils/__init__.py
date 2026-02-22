"""
utils/__init__.py
Shared helpers: LLM client, Tavily wrapper, date utilities.
Keeping them here avoids circular imports and lets you swap
implementations in one place (e.g., change model or search engine).
"""

from __future__ import annotations
from datetime import date
from typing import List, Optional

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()
# ─────────────────────────────────────────────
# LLM  (single shared instance)
# ─────────────────────────────────────────────
# ChatOpenAI wraps the OpenAI API.
# gpt-4.1-mini: fast + cheap — good for structured outputs.
# swap model="gpt-4o" for higher quality if budget allows.
llm = ChatGroq(model_name="openai/gpt-oss-120b")
# llm = ChatGroq(model="llama-3.1-8b-instant")

# ─────────────────────────────────────────────
# WEB SEARCH helper
# ─────────────────────────────────────────────
def tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Calls Tavily (a search API designed for LLM agents) and returns
    a normalized list of dicts with consistent keys.

    Why normalize? Tavily's raw response has inconsistent field names
    across result types; normalizing makes the research_node simpler.
    """
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})


    # Extract the actual results list from the response dict
    if isinstance(results, dict):
        results = results.get("results", [])
    print(f"\n\n Result of query {query} : \n {results}\n\n")
    normalized: List[dict] = []
    for r in (results or []):
        if not isinstance(r, dict):
            continue
        normalized.append({
            "title":        r.get("title") or "",
            "url":          r.get("url") or "",
            "snippet":      r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source":       r.get("source"),
        })
    return normalized


# ─────────────────────────────────────────────
# DATE helper
# ─────────────────────────────────────────────
def iso_to_date(s: Optional[str]) -> Optional[date]:
    """
    Safely parse an ISO date string like "2026-02-19".
    Returns None if the string is empty or malformed — prevents crashes
    when Tavily returns inconsistent date formats.
    """
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])  # take first 10 chars: "YYYY-MM-DD"
    except Exception:
        return None
