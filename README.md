# 🏥 MediGuide AI — Intelligent Healthcare Chatbot

A Streamlit-based intelligent chatbot built for the NeoStats AI Engineer case study.
It combines RAG, live web search, and multi-provider LLM support to answer healthcare questions accurately.

## 🚀 Features

| Feature | Description |
|---|---|
| **RAG Integration** | Upload PDFs/TXT/DOCX and chat over your own documents using FAISS vector search |
| **Live Web Search** | Automatically falls back to Serper/Tavily web search for current information |
| **Multi-LLM Support** | Switch between OpenAI, Groq (Llama 3), and Google Gemini |
| **Response Modes** | Toggle between Concise (3–4 sentences) and Detailed (structured, in-depth) |
| **Source Attribution** | Every response shows which KB chunks or web results were used |

## 📁 Project Structure

```
project/
├── config/
│   └── config.py         ← All API keys and settings (use .env)
├── models/
│   ├── llm.py            ← LLM wrappers (OpenAI / Groq / Gemini)
│   └── embeddings.py     ← Embedding models (HuggingFace / OpenAI)
├── utils/
│   ├── rag_utils.py      ← Document loading, chunking, FAISS vector store
│   ├── web_search.py     ← Serper & Tavily web search integration
│   └── chat_utils.py     ← Prompt building, history trimming, source formatting
├── data/                 ← Persisted vector store (auto-created)
├── app.py                ← Main Streamlit UI
└── requirements.txt
```

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/mediguide-ai.git
cd mediguide-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys to .env
cp .env.example .env
# Edit .env with your keys

# 5. Run the app
streamlit run app.py
```

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
GOOGLE_GEMINI_API_KEY=your_gemini_key
SERPER_API_KEY=your_serper_key
TAVILY_API_KEY=your_tavily_key
```

> ⚠️ Never commit your `.env` file. It is listed in `.gitignore`.

## 🌐 Deployment

Deployed on Streamlit Cloud: **[INSERT DEPLOYMENT LINK]**

## 📜 License
MIT
