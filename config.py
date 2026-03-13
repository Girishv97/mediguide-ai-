"""
config/config.py
Central configuration file for all API keys and application settings.
DO NOT commit real API keys to version control.
Use environment variables or a .env file for production deployments.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# ─── LLM Provider API Keys ────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
GOOGLE_GEMINI_API_KEY: str = os.getenv("GOOGLE_GEMINI_API_KEY", "your-gemini-api-key-here")

# ─── Web Search API Keys ──────────────────────────────────────────────────────
SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "your-serper-api-key-here")
# Alternative: Tavily Search (great for LLM agents)
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "your-tavily-api-key-here")

# ─── Embedding / RAG Settings ─────────────────────────────────────────────────
# Embedding model: "openai" | "huggingface" (local, free)
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Vector store settings
VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "data/vector_store")
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
TOP_K_RESULTS: int = 4

# ─── Application Settings ─────────────────────────────────────────────────────
APP_TITLE: str = "MediGuide AI — Intelligent Health Assistant"
APP_ICON: str = "🏥"
MAX_HISTORY_LENGTH: int = 20  # Max messages to retain in memory

# ─── LLM Default Settings ─────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER: str = "groq"           # "openai" | "groq" | "gemini"
DEFAULT_LLM_MODEL: str = "llama3-8b-8192"    # Groq fast model
TEMPERATURE: float = 0.4
MAX_TOKENS: int = 2048

# ─── Response Mode Prompts ────────────────────────────────────────────────────
CONCISE_INSTRUCTION: str = (
    "Respond concisely. Keep answers under 3–4 sentences unless explicitly asked for more. "
    "Be direct, clear, and avoid unnecessary elaboration."
)

DETAILED_INSTRUCTION: str = (
    "Respond in detail. Provide thorough, structured explanations with relevant context. "
    "Use headings, bullet points, or numbered lists where appropriate to improve readability."
)

# ─── System Prompt (Domain: Healthcare Q&A) ───────────────────────────────────
BASE_SYSTEM_PROMPT: str = """
You are MediGuide AI, an intelligent healthcare assistant designed to help users understand 
medical conditions, symptoms, medications, wellness tips, and general health information.

Guidelines:
- Always remind users that your information is educational and not a substitute for professional medical advice.
- When referencing documents from the knowledge base, cite the source clearly.
- Be empathetic, clear, and accurate.
- For emergencies, always direct users to call emergency services immediately.
- Do not diagnose or prescribe — educate and guide.
"""
