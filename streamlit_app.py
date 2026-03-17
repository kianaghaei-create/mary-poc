"""
MARY – Metodstöd AI
Streamlit-app för PoC: RAG på MARY-metoden med källhänvisningar.
"""

import json
import os
from pathlib import Path

import chromadb
import streamlit as st
from openai import OpenAI

# ── Sidinställningar ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MARY – Metodstöd AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background: #f4f1ec; }
  .block-container { padding-top: 1.5rem; max-width: 860px; }

  .hero {
    background: linear-gradient(135deg, #2d5a4f, #3a7a6a);
    color: white; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
  }
  .hero h1 { font-size: 1.6rem; margin: 0 0 6px; }
  .hero p  { font-size: 0.92rem; opacity: 0.85; margin: 0; line-height: 1.6; }

  .answer-box {
    background: white; border-radius: 14px; padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-top: 16px;
    font-size: 0.92rem; line-height: 1.75; color: #2c2c2c;
  }
  .source-box {
    background: #f4f1ec; border-radius: 12px; padding: 16px; margin-top: 12px;
  }
  .source-row {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 7px 0; border-bottom: 1px solid #e8e4dc; font-size: 0.83rem;
  }
  .source-row:last-child { border: none; }
  .src-title { font-weight: 600; color: #2c2c2c; }
  .src-url   { color: #aaa; font-size: 0.72rem; font-family: monospace; word-break: break-all; }
  .badge-web  { background:#e8f0fe;color:#2c5fbd;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600; }
  .badge-pdf  { background:#fde8e8;color:#b03030;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600; }
  .badge-docx { background:#eafaf1;color:#1e6e41;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600; }

  .pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
  .pill {
    background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.3);
    color:white; border-radius:20px; padding:5px 14px;
    font-size:0.78rem; cursor:pointer;
  }
  div[data-testid="stButton"] button {
    background: #2d5a4f; color: white; border: none;
    border-radius: 30px; font-weight: 600; padding: 10px 26px;
  }
  div[data-testid="stButton"] button:hover { background: #224439; }

  .stTextArea textarea {
    border-radius: 10px; border: 1.5px solid #ddd;
    font-family: 'Inter', sans-serif; font-size: 0.9rem;
  }
  .stTextArea textarea:focus { border-color: #2d5a4f; }

  .stat-card {
    background: white; border-radius: 12px; padding: 16px 20px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05); text-align: center;
  }
  .stat-num { font-size: 1.8rem; font-weight: 700; color: #2d5a4f; }
  .stat-lbl { font-size: 0.78rem; color: #888; margin-top: 2px; }

  /* Dölj streamlit-branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Ladda data & bygg ChromaDB (cachat) ──────────────────────────────────────
@st.cache_resource(show_spinner="Laddar MARY-kunskapsbasen...")
def load_rag():
    chunks_path = Path("rag_data/chunks.json")
    if not chunks_path.exists():
        st.error("rag_data/chunks.json saknas. Kör scraper.py + ingest.py lokalt först.")
        st.stop()

    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    # In-memory ChromaDB (fungerar på Streamlit Cloud)
    client = chromadb.Client()
    try:
        client.delete_collection("mary_rag")
    except Exception:
        pass
    col = client.create_collection("mary_rag", metadata={"hnsw:space": "cosine"})

    batch = 100
    for i in range(0, len(chunks), batch):
        b = chunks[i:i + batch]
        col.add(
            documents=[c["text"] for c in b],
            metadatas=[{"source": c["source"], "title": c["title"], "type": c["type"]} for c in b],
            ids=[f"c_{i+j}" for j in range(len(b))]
        )
    return col, chunks


@st.cache_resource
def get_openai():
    key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
    if not key:
        st.error("⚠️ OPENAI_API_KEY saknas. Lägg till den i Streamlit Secrets.")
        st.stop()
    return OpenAI(api_key=key)


collection, all_chunks = load_rag()
openai_client = get_openai()

SYSTEM_PROMPT = """Du är ett AI-stöd för handledare och medarbetare som arbetar med MARY-metoden inom Svenska kyrkan.

Din uppgift är att svara på frågor om MARY-metoden baserat ENBART på det underlag du fått.

Regler:
1. Svara alltid på svenska.
2. Basera ALLA påståenden på det underlag du fått — hitta inte på.
3. Om underlaget inte räcker, säg: "Det finns inte tillräckligt med underlag för detta i vårt material."
4. Avsluta alltid med källhänvisningar i formatet: [Källa: titel]
5. Håll svaret konkret och praktiskt — handledaren ska kunna agera direkt.
6. Markera citat från källorna med citationstecken.

Format:
- Direkt svar (2-4 meningar)
- Fördjupning med punktlista om relevant
- Källhänvisningar sist
"""

EXAMPLES = [
    "Jag har en deltagare som har tappat motivationen. Vilket nästa steg i Mary-metoden föreslår du?",
    "Vilka delar i materialet kan användas för en trygg start de första 2 veckorna?",
    "Vad är kärnan i Mary-metoden? Sammanfatta i punkter.",
    "Hur kopplas MMM – Meningen med Mig – till motivation och vardagsprogression?",
    "Vilka BIP-indikatorer nämns i materialet och var?",
]

TYPE_LABELS = {
    "web": ("🌐", "Webb", "badge-web"),
    "web_pdf": ("📄", "PDF (webb)", "badge-pdf"),
    "local_pdf": ("📄", "PDF (lokalt)", "badge-pdf"),
    "local_docx": ("📝", "Word-dok", "badge-docx"),
}

# ── RAG-funktion ──────────────────────────────────────────────────────────────
def ask_rag(question: str, top_k: int = 6):
    results = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    dists  = results["distances"][0]

    context_parts, sources, seen = [], [], set()
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        context_parts.append(
            f"[Underlag {i+1}] Titel: {meta['title']}\nKälla: {meta['source']}\n\n{doc}"
        )
        if meta["source"] not in seen:
            seen.add(meta["source"])
            sources.append(meta)

    context = "\n---\n".join(context_parts)
    user_msg = f"Fråga: {question}\n\nUnderlag:\n{context}\n\nSvara baserat på underlaget. Inkludera källhänvisningar."

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1400,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return response.choices[0].message.content, sources, len(docs)


# ── Layout ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌿 MARY – Metodstöd AI</h1>
  <p>Ställ en fråga om MARY-metoden. Svaret hämtas från lokala dokument och officiella webbsidor.<br>
     Varje svar kommer med källhänvisning — AI:n hittar inte på.</p>
</div>
""", unsafe_allow_html=True)

# Statistik-rad
c1, c2, c3 = st.columns(3)
unique_sources = len(set(c["source"] for c in all_chunks))
with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-num">{len(all_chunks)}</div><div class="stat-lbl">Textavsnitt indexerade</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-num">{unique_sources}</div><div class="stat-lbl">Unika källor</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-num">GPT-4o</div><div class="stat-lbl">Genereringsmodell</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Exempelfrågor
st.markdown("**Prova en exempelfråga:**")
cols = st.columns(len(EXAMPLES))
for i, (col, ex) in enumerate(zip(cols, EXAMPLES)):
    short = ex.split(".")[0][:30] + "…"
    if col.button(short, key=f"ex_{i}"):
        st.session_state["question"] = ex

# Frågefält
question = st.text_area(
    "Din fråga till MARY-materialet",
    value=st.session_state.get("question", ""),
    height=110,
    placeholder="T.ex. 'Hur anpassar vi upplägget för en deltagare som ofta uteblir?'",
    key="question_input",
)

ask_col, hint_col = st.columns([1, 5])
run = ask_col.button("🔍 Fråga MARY-metoden", use_container_width=True)

if run or (question and st.session_state.get("question") and st.session_state.get("auto_run")):
    q = question.strip()
    if q:
        with st.spinner("Söker i materialet…"):
            answer, sources, chunks_used = ask_rag(q)

        # Svar
        st.markdown(f'<div class="answer-box">{answer.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        # Källor
        src_rows = ""
        for s in sources:
            icon, label, badge_cls = TYPE_LABELS.get(s["type"], ("📎", s["type"], "badge-docx"))
            src_rows += f"""
            <div class="source-row">
              <span>{icon}</span>
              <div>
                <div class="src-title">{s['title']} <span class="{badge_cls}">{label}</span></div>
                <div class="src-url">{s['source']}</div>
              </div>
            </div>"""

        st.markdown(f"""
        <div class="source-box">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;color:#888;margin-bottom:10px;">
            📚 Källor — {len(sources)} dokument · {chunks_used} textavsnitt analyserade
          </div>
          {src_rows}
        </div>
        """, unsafe_allow_html=True)

        st.session_state["auto_run"] = False

# Auto-kör om exempelfråga klickades
if "question" in st.session_state and st.session_state.get("question"):
    st.session_state["auto_run"] = True

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:0.75rem;color:#bbb;">MARY Metodstöd AI · PoC · Svenska kyrkan · 2026</div>',
    unsafe_allow_html=True,
)
