"""
MARY RAG – API
FastAPI-server som tar emot en fråga och returnerar svar med källhänvisningar.
Använder ChromaDB för retrieval och OpenAI GPT-4o för generering.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import chromadb
from openai import OpenAI

CHROMA_DIR = str(Path("rag_data") / "chroma_db")
GPT_MODEL = "gpt-4o"
TOP_K = 6  # antal chunks att hämta

app = FastAPI(title="MARY RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import FileResponse, RedirectResponse

@app.get("/")
def root():
    return FileResponse("index.html")

# Serve static assets
app.mount("/static", StaticFiles(directory="."), name="static")

# Initiera klienter
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection("mary_rag")
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """Du är ett AI-stöd för handledare och medarbetare som arbetar med MARY-metoden inom Svenska kyrkan.

Din uppgift är att svara på frågor om MARY-metoden baserat ENBART på det underlag du fått.

Regler:
1. Svara alltid på svenska.
2. Basera ALLA påståenden på det underlag du fått — hitta inte på.
3. Om underlaget inte räcker för att svara, säg det tydligt: "Det finns inte tillräckligt med underlag för detta i vårt material."
4. Avsluta alltid med källhänvisningar i formatet: [Källa: titel, typ]
5. Håll svaret konkret och praktiskt — handledaren ska kunna agera direkt.
6. Markera citat från källorna med citationstecken.

Format för svar:
- Börja med ett direkt svar (2-4 meningar)
- Eventuell fördjupning med punktlista
- Källhänvisningar sist
"""

class QueryRequest(BaseModel):
    question: str
    top_k: int = TOP_K

class Source(BaseModel):
    title: str
    source: str
    type: str
    excerpt: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    chunks_used: int

@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": collection.count()}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(400, "Frågan får inte vara tom.")

    # 1. Hämta relevanta chunks från ChromaDB
    results = collection.query(
        query_texts=[req.question],
        n_results=min(req.top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not docs:
        raise HTTPException(404, "Inga relevanta chunks hittades.")

    # 2. Bygg kontext för Claude
    context_parts = []
    sources = []
    seen_sources = set()

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        relevance = round((1 - dist) * 100, 1)
        context_parts.append(
            f"[Underlag {i+1}] Titel: {meta['title']}\nKälla: {meta['source']}\n\n{doc}\n"
        )
        source_key = meta["source"]
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(Source(
                title=meta["title"],
                source=meta["source"],
                type=meta["type"],
                excerpt=doc[:200] + "..." if len(doc) > 200 else doc
            ))

    context = "\n---\n".join(context_parts)

    # 3. Generera svar med GPT-4o
    user_message = f"""Fråga från handledare: {req.question}

Tillgängligt underlag:
{context}

Svara på frågan baserat på ovanstående underlag. Inkludera källhänvisningar."""

    response = openai_client.chat.completions.create(
        model=GPT_MODEL,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.choices[0].message.content

    return QueryResponse(
        answer=answer,
        sources=sources,
        chunks_used=len(docs)
    )

@app.get("/stats")
def stats():
    """Statistik om indexet."""
    return {
        "total_chunks": collection.count(),
        "collection": "mary_rag",
        "model": GPT_MODEL,
        "top_k_default": TOP_K
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
