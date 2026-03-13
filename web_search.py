"""
utils/web_search.py
Live web search integration using Serper.dev (Google Search API).
Falls back to Tavily if Serper is unavailable.

Both services offer free tiers suitable for prototyping.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Serper Search ────────────────────────────────────────────────────────────

def search_serper(query: str, num_results: int = 5, api_key: Optional[str] = None) -> list[dict]:
    """
    Perform a real-time web search using Serper.dev (Google Search wrapper).

    Args:
        query: Search query string.
        num_results: Number of results to return.
        api_key: Optional API key override; falls back to config.

    Returns:
        List of dicts with keys: "title", "link", "snippet"
    """
    try:
        import requests
        from config.config import SERPER_API_KEY

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key or SERPER_API_KEY,
            "Content-Type": "application/json",
        }
        payload = {"q": query, "num": num_results}

        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        data = response.json()
        results = []

        for item in data.get("organic", [])[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
            )
        return results

    except Exception as exc:
        logger.error("Serper search error: %s", exc)
        return []


# ─── Tavily Search (Alternative) ──────────────────────────────────────────────

def search_tavily(query: str, num_results: int = 5, api_key: Optional[str] = None) -> list[dict]:
    """
    Perform a real-time web search using Tavily (AI-optimised search API).

    Args:
        query: Search query string.
        num_results: Number of results to return.
        api_key: Optional API key override; falls back to config.

    Returns:
        List of dicts with keys: "title", "link", "snippet"
    """
    try:
        from tavily import TavilyClient
        from config.config import TAVILY_API_KEY

        client = TavilyClient(api_key=api_key or TAVILY_API_KEY)
        response = client.search(query=query, max_results=num_results)

        results = []
        for item in response.get("results", []):
            results.append(
                {
                    "title": item.get("title", ""),
                    "link": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
            )
        return results

    except Exception as exc:
        logger.error("Tavily search error: %s", exc)
        return []


# ─── Unified Search Dispatcher ────────────────────────────────────────────────

def web_search(
    query: str,
    num_results: int = 5,
    provider: str = "serper",
) -> list[dict]:
    """
    Unified web search interface.

    Args:
        query: Search query.
        num_results: Number of results to return.
        provider: "serper" | "tavily"

    Returns:
        List of result dicts: {"title", "link", "snippet"}
    """
    provider = provider.lower().strip()

    if provider == "serper":
        return search_serper(query, num_results)
    elif provider == "tavily":
        return search_tavily(query, num_results)
    else:
        raise ValueError(f"Unsupported search provider '{provider}'. Choose: serper, tavily.")


# ─── Context Builder ──────────────────────────────────────────────────────────

def build_search_context(results: list[dict], query: str) -> str:
    """
    Format web search results into a structured context block for the LLM prompt.

    Args:
        results: Output from web_search().
        query: The original search query (for labelling).

    Returns:
        Formatted context string.
    """
    if not results:
        return ""

    lines = [f"### Live Web Search Results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"**[{i}] {r['title']}**\nURL: {r['link']}\n{r['snippet']}\n"
        )
    return "\n".join(lines)


# ─── Query Decision Helper ────────────────────────────────────────────────────

def should_use_web_search(user_query: str, rag_results: list[dict], min_rag_score: float = 0.4) -> bool:
    """
    Decide whether to fall back to web search based on RAG result quality.

    Args:
        user_query: The user's message.
        rag_results: Retrieved RAG chunks with scores.
        min_rag_score: Minimum cosine similarity score to trust RAG results.

    Returns:
        True if web search should be triggered.
    """
    # Always search for clearly time-sensitive queries
    time_keywords = ["latest", "recent", "today", "current", "news", "update", "2024", "2025"]
    if any(kw in user_query.lower() for kw in time_keywords):
        return True

    # Fall back to search if RAG confidence is low
    if not rag_results or rag_results[0].get("score", 0) < min_rag_score:
        return True

    return False
