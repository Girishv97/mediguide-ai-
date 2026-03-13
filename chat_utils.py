"""
utils/chat_utils.py
Chat history management and system prompt construction utilities.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def trim_history(messages: list[dict], max_length: int = 20) -> list[dict]:
    """
    Trim the chat history to avoid exceeding context window limits.
    Keeps the most recent `max_length` messages.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        max_length: Maximum number of messages to retain.

    Returns:
        Trimmed message list.
    """
    try:
        if len(messages) > max_length:
            return messages[-max_length:]
        return messages
    except Exception as exc:
        logger.error("History trim error: %s", exc)
        return messages


def build_system_prompt(
    base_prompt: str,
    response_mode: str = "detailed",
    rag_context: Optional[str] = None,
    search_context: Optional[str] = None,
) -> str:
    """
    Assemble the full system prompt by combining:
    - Base persona/role instructions
    - Response mode instructions (concise vs detailed)
    - RAG context (if any relevant documents were retrieved)
    - Web search context (if live search was performed)

    Args:
        base_prompt: Core system persona and rules.
        response_mode: "concise" | "detailed"
        rag_context: Formatted knowledge base excerpts string.
        search_context: Formatted web search results string.

    Returns:
        Complete system prompt string ready for the LLM.
    """
    try:
        from config.config import CONCISE_INSTRUCTION, DETAILED_INSTRUCTION

        parts = [base_prompt.strip()]

        # Append response mode instruction
        if response_mode.lower() == "concise":
            parts.append(f"\n\n**Response Style:**\n{CONCISE_INSTRUCTION}")
        else:
            parts.append(f"\n\n**Response Style:**\n{DETAILED_INSTRUCTION}")

        # Append RAG knowledge base context
        if rag_context:
            parts.append(
                "\n\n**Use the following knowledge base excerpts to answer accurately.**\n"
                "Cite the source name in your response when referencing these excerpts.\n\n"
                + rag_context
            )

        # Append web search context
        if search_context:
            parts.append(
                "\n\n**The following live web search results may supplement your knowledge.**\n"
                "Prioritise recent, credible information from these results.\n\n"
                + search_context
            )

        return "\n".join(parts)

    except Exception as exc:
        logger.error("System prompt build error: %s", exc)
        return base_prompt


def format_sources(rag_results: list[dict], search_results: list[dict]) -> str:
    """
    Build a readable sources footnote to display below the assistant's reply.

    Args:
        rag_results: Retrieved RAG chunks.
        search_results: Web search result dicts.

    Returns:
        Markdown-formatted sources string, or empty string if no sources.
    """
    try:
        lines = []

        if rag_results:
            seen_sources = set()
            for r in rag_results:
                src = r.get("source", "")
                if src and src not in seen_sources:
                    lines.append(f"📄 **KB:** {src} (score: {r['score']:.2f})")
                    seen_sources.add(src)

        if search_results:
            for r in search_results:
                title = r.get("title", "Web result")
                link = r.get("link", "")
                if link:
                    lines.append(f"🌐 **Web:** [{title}]({link})")

        if not lines:
            return ""

        return "\n\n---\n**Sources:**\n" + "\n".join(lines)

    except Exception as exc:
        logger.error("Source formatting error: %s", exc)
        return ""
