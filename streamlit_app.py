import os, json, threading
from pathlib import Path
import streamlit as st
import chromadb
from openai import OpenAI

# ── Start FastAPI in background thread so index.html can call localhost:8000 ──
@st.cache_resource
def start_api_server():
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    _col = None

    def get_col():
        nonlocal _col
        if _col is None:
            client = chromadb.Client()
            try:
                client.delete_collection("mary_api")
            except Exception:
                pass
            col = client.create_collection("mary_api", metadata={"hnsw:space": "cosine"})
            with open(Path("rag_data/chunks.json"), encoding="utf-8") as f:
                chunks = json.load(f)
            for i in range(0, len(chunks), 100):
                b = chunks[i:i+100]
                col.add(
                    documents=[c["text"] for c in b],
                    metadatas=[{"source": c["source"], "title": c["title"], "type": c["type"]} for c in b],
                    ids=[f"a_{i+j}" for j in range(len(b))],
                )
            _col = col
        return _col

    class Q(BaseModel):
        question: str
        top_k: int = 6

    SYSTEM = """Du är ett AI-stöd för handledare som arbetar med MARY-metoden inom Svenska kyrkan.
Svara ENBART baserat på det underlag du fått. Svara på svenska. Avsluta med [Källa: titel]."""

    @api.post("/query")
    def query(req: Q):
        col = get_col()
        key = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        results = col.query(query_texts=[req.question], n_results=min(req.top_k, col.count()),
                            include=["documents", "metadatas", "distances"])
        docs, metas = results["documents"][0], results["metadatas"][0]
        context = "\n---\n".join(
            f"[Underlag {i+1}] Titel: {m['title']}\nKälla: {m['source']}\n\n{d}"
            for i, (d, m) in enumerate(zip(docs, metas))
        )
        oai = OpenAI(api_key=key)
        resp = oai.chat.completions.create(
            model="gpt-4o", max_tokens=1200,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Fråga: {req.question}\n\nUnderlag:\n{context}"},
            ],
        )
        seen, sources = set(), []
        for d, m in zip(docs, metas):
            if m["source"] not in seen:
                seen.add(m["source"])
                sources.append({"title": m["title"], "source": m["source"],
                                 "type": m["type"], "excerpt": d[:200]})
        return {"answer": resp.choices[0].message.content, "sources": sources, "chunks_used": len(docs)}

    @api.get("/health")
    def health():
        return {"status": "ok"}

    t = threading.Thread(
        target=lambda: uvicorn.run(api, host="0.0.0.0", port=8000, log_level="error"),
        daemon=True,
    )
    t.start()
    return True

start_api_server()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MARY – Min Resa & Metodstöd",
    page_icon="🌿",
    layout="wide",
)

# ── Load OpenAI key ───────────────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

# ── Build ChromaDB in-memory from chunks.json (cached) ───────────────────────
@st.cache_resource(show_spinner="Bygger kunskapsbas från MARY-material…")
def build_collection():
    chunks_path = Path("rag_data/chunks.json")
    client = chromadb.Client()
    try:
        client.delete_collection("mary_rag")
    except Exception:
        pass
    col = client.create_collection("mary_rag", metadata={"hnsw:space": "cosine"})
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    for i in range(0, len(chunks), 100):
        b = chunks[i:i + 100]
        col.add(
            documents=[c["text"] for c in b],
            metadatas=[{"source": c["source"], "title": c["title"], "type": c["type"]} for c in b],
            ids=[f"c_{i+j}" for j in range(len(b))],
        )
    return col

collection = build_collection()

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Du är ett AI-stöd för handledare och medarbetare som arbetar med MARY-metoden inom Svenska kyrkan.

Din uppgift är att svara på frågor om MARY-metoden baserat ENBART på det underlag du fått.

Regler:
1. Svara alltid på svenska.
2. Basera ALLA påståenden på det underlag du fått — hitta inte på.
3. Om underlaget inte räcker, säg det tydligt och föreslå vad som behöver utredas vidare.
4. Avsluta alltid med källhänvisningar i formatet: [Källa: titel]
5. Håll svaret konkret och praktiskt — handledaren ska kunna agera direkt.
"""

def query_rag(question: str, top_k: int = 6) -> dict:
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    context = "\n---\n".join(
        f"[Underlag {i+1}] Titel: {m['title']}\nKälla: {m['source']}\n\n{doc}"
        for i, (doc, m) in enumerate(zip(docs, metas))
    )

    oai = OpenAI(api_key=OPENAI_API_KEY)
    response = oai.chat.completions.create(
        model="gpt-4o",
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fråga: {question}\n\nUnderlag:\n{context}"},
        ],
    )
    answer = response.choices[0].message.content

    seen, sources = set(), []
    for doc, m in zip(docs, metas):
        if m["source"] not in seen:
            seen.add(m["source"])
            sources.append({
                "title": m["title"],
                "source": m["source"],
                "type": m["type"],
                "excerpt": doc[:220] + "…" if len(doc) > 220 else doc,
            })
    return {"answer": answer, "sources": sources}

# ── Read HTML shell ───────────────────────────────────────────────────────────
html_shell = Path("index.html").read_text(encoding="utf-8")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #f4f1ec; }
  [data-testid="stHeader"] { background: transparent; }
  section[data-testid="stSidebar"] { display: none; }

  div[data-baseweb="tab-list"] {
    background: #2d5a4f !important;
    border-radius: 0 !important;
    padding: 0 24px !important;
    gap: 0 !important;
  }
  div[data-baseweb="tab"] {
    color: rgba(255,255,255,0.65) !important;
    font-weight: 600 !important;
    padding: 14px 24px !important;
    border-radius: 0 !important;
    border-bottom: 3px solid transparent !important;
  }
  div[data-baseweb="tab"][aria-selected="true"] {
    color: white !important;
    border-bottom: 3px solid #a8d5c2 !important;
    background: transparent !important;
  }
  div[data-baseweb="tab-highlight"] { display: none !important; }
  div[data-baseweb="tab-border"] { display: none !important; }

  .chat-user {
    background: #2d5a4f; color: white;
    padding: 12px 16px; border-radius: 16px 16px 4px 16px;
    margin: 8px 0 8px auto; max-width: 75%; width: fit-content;
    font-size: 0.92rem;
  }
  .chat-ai {
    background: white; color: #2c2c2c;
    padding: 14px 18px; border-radius: 16px 16px 16px 4px;
    margin: 8px auto 8px 0; max-width: 85%;
    font-size: 0.9rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }
  .source-block {
    background: #f9f7f4; border-radius: 10px; padding: 10px 14px;
    margin-top: 8px; border-left: 3px solid #a8d5c2;
  }
  .source-block .s-title { font-size: 0.82rem; font-weight: 700; color: #2d5a4f; }
  .source-block .s-excerpt { font-size: 0.77rem; color: #666; margin-top: 3px; }
  .source-block .s-url { font-size: 0.7rem; color: #aaa; margin-top: 2px; word-break: break-all; }

  .stButton button {
    background: white !important;
    border: 1.5px solid #c8ddd8 !important;
    color: #2d5a4f !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    padding: 6px 14px !important;
    white-space: normal !important;
    text-align: left !important;
  }
  .stButton button:hover {
    background: #2d5a4f !important;
    color: white !important;
    border-color: #2d5a4f !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_dashboard, tab_rag = st.tabs(["📋  Dashboard – Min Resa", "🤖  Metodstöd AI"])

# ── TAB 1: Dashboard ──────────────────────────────────────────────────────────
with tab_dashboard:
    st.components.v1.html(html_shell, height=1000, scrolling=True)

# ── TAB 2: RAG chat ───────────────────────────────────────────────────────────
with tab_rag:
    col_chat, col_sources = st.columns([3, 2])

    with col_chat:
        st.markdown("### Fråga om MARY-metoden")
        st.caption(f"Kunskapsbas: {collection.count()} indexerade stycken från MARY-dokumentationen")

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "last_sources" not in st.session_state:
            st.session_state.last_sources = []

        if not st.session_state.messages:
            st.markdown("**Föreslagna frågor:**")
            suggestions = [
                "Vad är kärnan i MARY-metoden?",
                "Hur stöttar jag en deltagare som tappat motivationen?",
                "Vilka BIP-indikatorer ingår i MARY?",
                "Hur ser de första 2 veckorna ut för en ny deltagare?",
                "Vad är S:t Mary och hur skiljer det sig från Maryplats?",
            ]
            cols = st.columns(2)
            for i, s in enumerate(suggestions):
                if cols[i % 2].button(s, key=f"sug_{i}"):
                    st.session_state._pending_question = s

        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_input(
                "Din fråga",
                placeholder="T.ex. Hur hanterar jag en deltagare som ofta uteblir?",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Skicka →", use_container_width=True)

        question = None
        if submitted and user_input.strip():
            question = user_input.strip()
        elif hasattr(st.session_state, "_pending_question"):
            question = st.session_state._pending_question
            del st.session_state._pending_question

        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            if not OPENAI_API_KEY:
                answer_data = {
                    "answer": "⚠️ OPENAI_API_KEY saknas. Lägg till den i Streamlit Secrets (Settings → Secrets).",
                    "sources": [],
                }
            else:
                with st.spinner("Söker i MARY-materialet…"):
                    answer_data = query_rag(question)
            st.session_state.messages.append({"role": "assistant", "content": answer_data["answer"]})
            st.session_state.last_sources = answer_data["sources"]
            st.rerun()

    with col_sources:
        st.markdown("### Källhänvisningar")
        if st.session_state.last_sources:
            st.caption(f"{len(st.session_state.last_sources)} unika källor använda i svaret")
            for s in st.session_state.last_sources:
                st.markdown(f"""
                <div class="source-block">
                  <div class="s-title">📄 {s['title']}</div>
                  <div class="s-excerpt">{s['excerpt']}</div>
                  <div class="s-url">{s['source']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<div style="color:#aaa;font-size:0.85rem;margin-top:20px;text-align:center;">'
                'Källorna till svaret visas här efter varje fråga.</div>',
                unsafe_allow_html=True,
            )

        if st.session_state.messages:
            st.write("")
            if st.button("🗑️ Rensa konversation", use_container_width=True):
                st.session_state.messages = []
                st.session_state.last_sources = []
                st.rerun()
