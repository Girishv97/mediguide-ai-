"""
models/llm.py
LLM provider wrappers for OpenAI, Groq, and Google Gemini.
All models share a common interface: get_llm_response(messages, system_prompt, ...) -> str
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ─── OpenAI ───────────────────────────────────────────────────────────────────

def get_openai_response(
    messages: list[dict],
    system_prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> str:
    """
    Call the OpenAI Chat Completions API and return the assistant's reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts (chat history).
        system_prompt: Instruction string prepended as the system message.
        model: OpenAI model name.
        temperature: Sampling temperature (0–2).
        max_tokens: Maximum tokens in the response.
        api_key: Optional override; falls back to config.

    Returns:
        The assistant's response text.
    """
    try:
        from openai import OpenAI
        from config.config import OPENAI_API_KEY

        client = OpenAI(api_key=api_key or OPENAI_API_KEY)

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("OpenAI API error: %s", exc)
        raise RuntimeError(f"OpenAI error: {exc}") from exc


# ─── Groq ─────────────────────────────────────────────────────────────────────

def get_groq_response(
    messages: list[dict],
    system_prompt: str,
    model: str = "llama3-8b-8192",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> str:
    """
    Call the Groq Chat Completions API (OpenAI-compatible) and return the reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        system_prompt: System instruction string.
        model: Groq model name (e.g., llama3-8b-8192, mixtral-8x7b-32768).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
        api_key: Optional override; falls back to config.

    Returns:
        The assistant's response text.
    """
    try:
        from groq import Groq
        from config.config import GROQ_API_KEY

        client = Groq(api_key=api_key or GROQ_API_KEY)

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        completion = client.chat.completions.create(
            model=model,
            messages=full_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content.strip()

    except Exception as exc:
        logger.error("Groq API error: %s", exc)
        raise RuntimeError(f"Groq error: {exc}") from exc


# ─── Google Gemini ────────────────────────────────────────────────────────────

def get_gemini_response(
    messages: list[dict],
    system_prompt: str,
    model: str = "gemini-1.5-flash",
    temperature: float = 0.4,
    max_tokens: int = 2048,
    api_key: Optional[str] = None,
) -> str:
    """
    Call the Google Generative AI (Gemini) API and return the reply.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        system_prompt: System instruction string (injected as first user turn).
        model: Gemini model name.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        api_key: Optional override; falls back to config.

    Returns:
        The assistant's response text.
    """
    try:
        import google.generativeai as genai
        from config.config import GOOGLE_GEMINI_API_KEY

        genai.configure(api_key=api_key or GOOGLE_GEMINI_API_KEY)

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        gemini_model = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_prompt,
            generation_config=generation_config,
        )

        # Convert OpenAI-style history to Gemini content format
        gemini_history = []
        for msg in messages[:-1]:   # All but the last message go into history
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = gemini_model.start_chat(history=gemini_history)
        response = chat.send_message(messages[-1]["content"])
        return response.text.strip()

    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        raise RuntimeError(f"Gemini error: {exc}") from exc


# ─── Unified Dispatcher ───────────────────────────────────────────────────────

def get_llm_response(
    messages: list[dict],
    system_prompt: str,
    provider: str = "groq",
    model: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 2048,
) -> str:
    """
    Unified interface to call any supported LLM provider.

    Args:
        messages: Chat history as list of role/content dicts.
        system_prompt: System-level instruction for the model.
        provider: "openai" | "groq" | "gemini"
        model: Model name (uses provider default if None).
        temperature: Sampling temperature.
        max_tokens: Max tokens in the response.

    Returns:
        The model's text response.

    Raises:
        ValueError: If an unsupported provider is specified.
    """
    from config.config import DEFAULT_LLM_MODEL

    provider = provider.lower().strip()

    if provider == "openai":
        return get_openai_response(
            messages, system_prompt,
            model=model or "gpt-4o-mini",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "groq":
        return get_groq_response(
            messages, system_prompt,
            model=model or "llama3-8b-8192",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "gemini":
        return get_gemini_response(
            messages, system_prompt,
            model=model or "gemini-1.5-flash",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported provider '{provider}'. Choose from: openai, groq, gemini."
        )
