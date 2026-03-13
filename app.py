"""
app.py
MediGuide AI — Intelligent Healthcare Chatbot
Main Streamlit application entry point.

Features:
- Multi-provider LLM support (OpenAI / Groq / Gemini)
- RAG integration with document upload
- Live web search fallback
- Concise / Detailed response modes
- Persistent chat history with source attribution
"""

import os
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ─── Import project modules ───────────────────────────────────────────────────
try:
    from config.config import (
        APP_TITLE,
        APP_ICON,
        BASE_SYSTEM_PROMPT,
        DEFAULT_LLM_PROVIDER,
        DEFAULT_LLM_MODEL,
        TEMPERATURE,
        MAX_TOKENS,
        MAX_HISTORY_LENGTH,
        EMBEDDING_PROVIDER,
        TOP_K_RESULTS,
        VECTOR_STORE_PATH,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    from models.llm import get_llm_response
    from models.embeddings import get_embeddings
    from utils.rag_utils import VectorStore, build_rag_context
    from utils.web_search import web_search, build_search_context, should_use_web_search
    from utils.chat_utils import trim_history, build_system_prompt, format_sources
except ImportError as e:
    st.error(f"Import error: {e}. Ensure all dependencies are installed.")
    st.stop()


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1a73e8;}
    .sub-header  {font-size: 0.95rem; color: #5f6368; margin-top: -0.5rem;}
    .source-box  {
        background: #f1f8ff; border-left: 3px solid #1a73e8;
        padding: 0.5rem 1rem; border-radius: 4px;
        font-size: 0.82rem; color: #444;
    }
    .status-pill {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.75rem; font-weight: 600;
    }
    .pill-rag    {background:#e8f5e9; color:#2e7d32;}
    .pill-web    {background:#fff3e0; color:#e65100;}
    .pill-llm    {background:#e3f2fd; color:#1565c0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ─── Session State Initialisation ────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "messages": [],                          # Chat history
        "vector_store": None,                    # VectorStore instance
        "rag_enabled": False,                    # RAG toggle
        "web_search_enabled": True,              # Web search toggle
        "response_mode": "Detailed",             # "Concise" | "Detailed"
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "llm_model": DEFAULT_LLM_MODEL,
        "show_sources": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_ICON} {APP_TITLE}")
    st.markdown("---")

    # ── LLM Provider ──────────────────────────────────────────────────────────
    st.subheader("🤖 LLM Settings")

    provider = st.selectbox(
        "Provider",
        options=["groq", "openai", "gemini"],
        index=["groq", "openai", "gemini"].index(st.session_state.llm_provider),
        key="provider_select",
    )
    st.session_state.llm_provider = provider

    model_options = {
        "groq":   ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768"],
        "openai": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    }
    selected_models = model_options[provider]
    idx = 0
    if st.session_state.llm_model in selected_models:
        idx = selected_models.index(st.session_state.llm_model)

    model = st.selectbox("Model", options=selected_models, index=idx)
    st.session_state.llm_model = model

    st.markdown("---")

    # ── Response Mode ─────────────────────────────────────────────────────────
    st.subheader("💬 Response Mode")
    response_mode = st.radio(
        "Select mode:",
        options=["Concise", "Detailed"],
        index=0 if st.session_state.response_mode == "Concise" else 1,
        horizontal=True,
    )
    st.session_state.response_mode = response_mode
    if response_mode == "Concise":
        st.caption("Short, summarised replies (3–4 sentences)")
    else:
        st.caption("In-depth, structured responses with context")

    st.markdown("---")

    # ── Feature Toggles ───────────────────────────────────────────────────────
    st.subheader("⚙️ Features")

    web_enabled = st.toggle(
        "🌐 Live Web Search",
        value=st.session_state.web_search_enabled,
        help="Automatically search the web when knowledge base results are weak.",
    )
    st.session_state.web_search_enabled = web_enabled

    show_sources = st.toggle(
        "📎 Show Sources",
        value=st.session_state.show_sources,
        help="Display RAG and web search sources below each response.",
    )
    st.session_state.show_sources = show_sources

    st.markdown("---")

    # ── Document Upload (RAG) ─────────────────────────────────────────────────
    st.subheader("📂 Knowledge Base (RAG)")

    uploaded_files = st.file_uploader(
        "Upload documents (PDF, TXT, DOCX)",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx"],
    )

    if uploaded_files:
        if st.button("📥 Index Documents", use_container_width=True):
            _index_documents(uploaded_files)

    if st.session_state.vector_store and len(st.session_state.vector_store) > 0:
        count = len(st.session_state.vector_store)
        st.success(f"✅ {count} chunks indexed")
        st.session_state.rag_enabled = True

        if st.button("🗑️ Clear Knowledge Base", use_container_width=True):
            st.session_state.vector_store.clear()
            st.session_state.rag_enabled = False
            st.rerun()
    else:
        st.info("No documents indexed yet.")
        st.session_state.rag_enabled = False

    st.markdown("---")

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Document Indexing Function ───────────────────────────────────────────────
def _index_documents(files) -> None:
    """Save uploaded files to a temp directory and index them into the vector store."""
    import tempfile

    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStore(store_path=VECTOR_STORE_PATH)

    with st.spinner("Indexing documents... this may take a moment."):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                for f in files:
                    dest = os.path.join(tmpdir, f.name)
                    with open(dest, "wb") as out:
                        out.write(f.read())
                    paths.append(dest)

                added = st.session_state.vector_store.add_documents(
                    file_paths=paths,
                    chunk_size=CHUNK_SIZE,
                    overlap=CHUNK_OVERLAP,
                    embedding_provider=EMBEDDING_PROVIDER,
                )
                st.success(f"Indexed {added} new chunks from {len(files)} file(s).")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")
            logger.error("Document indexing error: %s", exc)


# ─── Main Chat Interface ───────────────────────────────────────────────────────
st.markdown(f'<div class="main-header">{APP_ICON} {APP_TITLE}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Your AI-powered health information assistant. '
    "Not a substitute for professional medical advice.</div>",
    unsafe_allow_html=True,
)
st.markdown("")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if stored
        if message.get("sources") and st.session_state.show_sources:
            st.markdown(
                f'<div class="source-box">{message["sources"]}</div>',
                unsafe_allow_html=True,
            )


# ─── Chat Input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything about health, symptoms, or wellness…"):

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Append to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── Pipeline ──────────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                rag_results: list[dict] = []
                search_results: list[dict] = []
                used_rag = False
                used_web = False

                # 1. RAG Retrieval
                if st.session_state.rag_enabled and st.session_state.vector_store:
                    rag_results = st.session_state.vector_store.search(
                        query=user_input,
                        top_k=TOP_K_RESULTS,
                        embedding_provider=EMBEDDING_PROVIDER,
                    )
                    if rag_results and rag_results[0]["score"] >= 0.35:
                        used_rag = True

                # 2. Web Search (if needed)
                if st.session_state.web_search_enabled and should_use_web_search(
                    user_input, rag_results
                ):
                    search_results = web_search(user_input, num_results=4)
                    if search_results:
                        used_web = True

                # 3. Build system prompt with all context
                rag_ctx = build_rag_context(rag_results) if used_rag else None
                search_ctx = build_search_context(search_results, user_input) if used_web else None

                system_prompt = build_system_prompt(
                    base_prompt=BASE_SYSTEM_PROMPT,
                    response_mode=st.session_state.response_mode,
                    rag_context=rag_ctx,
                    search_context=search_ctx,
                )

                # 4. Trim history and call LLM
                history = trim_history(
                    st.session_state.messages, max_length=MAX_HISTORY_LENGTH
                )

                reply = get_llm_response(
                    messages=history,
                    system_prompt=system_prompt,
                    provider=st.session_state.llm_provider,
                    model=st.session_state.llm_model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )

                # 5. Display reply
                st.markdown(reply)

                # 6. Build and display source attribution
                sources_text = format_sources(
                    rag_results if used_rag else [],
                    search_results if used_web else [],
                )
                if sources_text and st.session_state.show_sources:
                    st.markdown(
                        f'<div class="source-box">{sources_text}</div>',
                        unsafe_allow_html=True,
                    )

                # 7. Show pipeline status badges
                pills = []
                if used_rag:
                    pills.append('<span class="status-pill pill-rag">📄 RAG</span>')
                if used_web:
                    pills.append('<span class="status-pill pill-web">🌐 Web</span>')
                pills.append(
                    f'<span class="status-pill pill-llm">'
                    f'🤖 {st.session_state.llm_provider.upper()} · {st.session_state.llm_model}'
                    f'</span>'
                )
                st.markdown(" ".join(pills), unsafe_allow_html=True)

                # 8. Append to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": reply,
                        "sources": sources_text,
                    }
                )

            except RuntimeError as exc:
                error_msg = (
                    f"⚠️ Unable to generate a response: {exc}\n\n"
                    "Please check your API keys in `config/config.py`."
                )
                st.error(error_msg)
                logger.error("LLM call failed: %s", exc)
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                logger.exception("Unhandled exception in chat pipeline")
