"""
MARY-metoden scraper
Laddar ned HTML-sidor och PDF:er från publika källor + lokala filer.
Sparar allt som strukturerade text-chunks med metadata.
"""

import os, re, json, time, hashlib
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import pdfplumber
from docx import Document

OUTPUT_DIR = Path("rag_data")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MaryRAG/1.0)"}

URLS = [
    {"url": "https://www.svenskakyrkan.se/mary",                  "title": "Mary – övergripande ingångssida"},
    {"url": "https://www.svenskakyrkan.se/mary/metoden",          "title": "Marymetoden – central beskrivning"},
    {"url": "https://www.svenskakyrkan.se/mary/plats",            "title": "Maryplats – arbetsträning i församlingen"},
    {"url": "https://www.svenskakyrkan.se/mary/st-mary",          "title": "S:t Mary – självständiga verksamheter"},
    {"url": "https://www.svenskakyrkan.se/mary/om-stmary",        "title": "Om S:t Mary"},
    {"url": "https://www.svenskakyrkan.se/mary/utbildningar",     "title": "Maryutbildningar"},
    {"url": "https://www.skao.se/nyheter-nyhetsbrev-medlemstidning/medlemstidningen-ducatus/2025/marymetoden/", "title": "Marymetoden – ett stöd för livsförändring (SKAO)"},
    {"url": "https://www.svenskakyrkan.se/luleastift/st-mary",    "title": "S:t Mary Luleå stift"},
    # PDFs
    {"url": "https://www.svenskakyrkan.se/filer/500655/MARY-metoden%281%29.pdf",         "title": "MARY-metoden PDF-broschyr"},
    {"url": "https://www.svenskakyrkan.se/filer/500678/Utbildningar%20React%20MARY.pdf", "title": "Utbildningar React MARY PDF"},
    {"url": "https://www.svenskakyrkan.se/filer/1374643/SK22145_Vi%20skapar%20plats%20reviderad%20%20ah%2019%20okt.pdf", "title": "Vi skapar plats PDF"},
    {"url": "https://www.svenskakyrkan.se/filer/1147710/MARY%20infofolder%20till%20hemsidan%281%29.pdf", "title": "MARY infofolder"},
    {"url": "https://www.svenskakyrkan.se/Sve/Bin%C3%A4rfiler/Filer/Webbbroschyr%204%281%29.pdf", "title": "Webbbroschyr 4 – arbetsträning"},
    {"url": "https://www.mchs.se/download/18.55070bd218f39481be761a58/1715789022142/St%20Mary%20Slutrapport%20Lule%C3%A5%20stift%202020%20sammanhang%20och%20egenmakt.pdf", "title": "Slutrapport Luleå stift 2020"},
    {"url": "https://www.svenskakyrkan.se/filer/556588/Nr%2025-26%202022.pdf",           "title": "Mary-metoden sprids nationellt PDF"},
]

LOCAL_FILES = [
    "2 Mary En översikt 1.0 Intern (3).pdf",
    "3.1 Mary Checklista Gröna tråden 1.0 (2) (2).pdf",
    "4.1 Mary Checklista Röda tråden 1.0.pdf",
    "5.2 Mary Hållbara glasögon utskriftsversion 1.0.pdf",
    "6.3 Mary för församlingen 1.0.pdf",
    "IOP-avtal_mall_MARY (3).docx",
    "Marymanifestet En teologisk reflektion av begreppen att skapa, ge och ta plats (1) (1).pdf",
    "Publik dokumentation om MARY.docx",
    "Svenska kyrkan Mary.docx",
]

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Delar upp text i överlappande chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 100:  # skippa för korta chunks
            chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def scrape_html(url: str, title: str) -> list[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        # Ta bort navigation, footer, scripts
        for tag in soup(['nav', 'footer', 'script', 'style', 'header', 'aside']):
            tag.decompose()
        text = clean_text(soup.get_text(separator=' '))
        chunks = chunk_text(text)
        print(f"  ✅ HTML: {title} — {len(chunks)} chunks")
        return [{"text": c, "source": url, "title": title, "type": "web"} for c in chunks]
    except Exception as e:
        print(f"  ⚠️  HTML misslyckades: {title} — {e}")
        return []

def scrape_pdf_url(url: str, title: str) -> list[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        tmp = Path(f"/tmp/mary_{hashlib.md5(url.encode()).hexdigest()[:8]}.pdf")
        tmp.write_bytes(r.content)
        chunks = extract_pdf(tmp, title, source=url, source_type="web_pdf")
        tmp.unlink()
        return chunks
    except Exception as e:
        print(f"  ⚠️  PDF-url misslyckades: {title} — {e}")
        return []

def extract_pdf(path: Path, title: str, source: str = None, source_type: str = "local_pdf") -> list[dict]:
    source = source or str(path)
    try:
        with pdfplumber.open(path) as pdf:
            full_text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"
        full_text = clean_text(full_text)
        chunks = chunk_text(full_text)
        print(f"  ✅ PDF: {title} — {len(chunks)} chunks")
        return [{"text": c, "source": source, "title": title, "type": source_type} for c in chunks]
    except Exception as e:
        print(f"  ⚠️  PDF-extraktion misslyckades: {title} — {e}")
        return []

def extract_docx(path: Path, title: str) -> list[dict]:
    try:
        doc = Document(path)
        full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        full_text = clean_text(full_text)
        chunks = chunk_text(full_text)
        print(f"  ✅ DOCX: {title} — {len(chunks)} chunks")
        return [{"text": c, "source": str(path.name), "title": title, "type": "local_docx"} for c in chunks]
    except Exception as e:
        print(f"  ⚠️  DOCX-extraktion misslyckades: {title} — {e}")
        return []

def main():
    all_chunks = []

    print("\n📡 Skrapar webbsidor och PDF:er från nätet...\n")
    for item in URLS:
        url = item["url"]
        title = item["title"]
        if url.lower().endswith(".pdf"):
            chunks = scrape_pdf_url(url, title)
        else:
            chunks = scrape_html(url, title)
        all_chunks.extend(chunks)
        time.sleep(0.5)  # schysst mot servrar

    print(f"\n📁 Bearbetar lokala filer...\n")
    base = Path(__file__).parent
    for filename in LOCAL_FILES:
        path = base / filename
        if not path.exists():
            print(f"  ⚠️  Hittar inte: {filename}")
            continue
        if filename.endswith(".pdf"):
            chunks = extract_pdf(path, title=filename)
        elif filename.endswith(".docx") or filename.endswith(".dotx"):
            chunks = extract_docx(path, title=filename)
        all_chunks.extend(chunks)

    # Spara
    out = OUTPUT_DIR / "chunks.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Klar! {len(all_chunks)} chunks sparade → {out}")
    print(f"   Unika källor: {len(set(c['source'] for c in all_chunks))}")

if __name__ == "__main__":
    main()
