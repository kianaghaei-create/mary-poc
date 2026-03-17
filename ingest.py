"""
MARY RAG – Ingest
Läser chunks.json och bygger en ChromaDB-vektordatabas med Voyage-embeddings via Anthropic.
"""

import json, os
from pathlib import Path
import chromadb
import anthropic

RAG_DATA = Path("rag_data")
CHUNKS_FILE = RAG_DATA / "chunks.json"
CHROMA_DIR = str(RAG_DATA / "chroma_db")

EMBED_MODEL = "voyage-3"   # Anthropics embedding-modell via voyage
EMBED_BATCH = 32           # max per anrop

def get_embeddings(client: anthropic.Anthropic, texts: list[str]) -> list[list[float]]:
    """Hämtar embeddings i batchar via Anthropic Voyage."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        response = client.beta.messages.batches  # voyage via anthropic
        # Voyage direkt via anthropic client
        result = client._client.post(
            "https://api.voyageai.com/v1/embeddings",
            json={"input": batch, "model": EMBED_MODEL},
            headers={"Authorization": f"Bearer {os.environ.get('VOYAGE_API_KEY', os.environ.get('ANTHROPIC_API_KEY', ''))}"}
        )
        # Fallback: använd anthropic embeddings endpoint
        raise NotImplementedError("Använd voyage direkt")
    return all_embeddings

def get_embeddings_voyage(texts: list[str]) -> list[list[float]]:
    """Voyage embeddings (rekommenderat av Anthropic)."""
    import voyageai
    vo = voyageai.Client()
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i + EMBED_BATCH]
        result = vo.embed(batch, model=EMBED_MODEL)
        all_embeddings.extend(result.embeddings)
    return all_embeddings

def get_embeddings_simple(texts: list[str]) -> list[list[float]]:
    """
    Fallback: använd chromadb:s inbyggda sentence-transformer.
    Inget API-nyckel krävs — bra för lokal PoC.
    """
    # ChromaDB hanterar detta automatiskt när vi inte anger embedding_function
    # Returnerar None = chromadb använder sin default
    return None

def main():
    if not CHUNKS_FILE.exists():
        print("❌ chunks.json saknas — kör scraper.py först!")
        return

    with open(CHUNKS_FILE, encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"📦 Laddar {len(chunks)} chunks till ChromaDB...")

    # Använd ChromaDB med inbyggd embedding (sentence-transformers)
    # Inget API-nyckel krävs för PoC — byt till Voyage för produktion
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Ta bort gammal collection om den finns
    try:
        client.delete_collection("mary_rag")
    except Exception:
        pass

    collection = client.create_collection(
        name="mary_rag",
        metadata={"hnsw:space": "cosine"}
    )

    # Lägg till i batchar
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "title": c["title"], "type": c["type"]} for c in batch],
            ids=[f"chunk_{i + j}" for j, c in enumerate(batch)]
        )
        print(f"  ✅ Indexerade {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"\n✅ ChromaDB klar! {collection.count()} chunks indexerade → {CHROMA_DIR}")

if __name__ == "__main__":
    main()
