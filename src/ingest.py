import json
import time
from pathlib import Path
import pymupdf
import chromadb
from src.embed import embed_passages
from src.llm import ask_with_image

db = chromadb.PersistentClient(path='db')
col = db.get_or_create_collection('docs', metadata={'hnsw:space': 'cosine'})

CACHE_FILE = Path('cache/ocr.json')

OCR_PROMPT = (
    'Transcribe all the text on this page exactly as written. '
    'Keep tables as Markdown tables. Keep headings and numbering. '
    'The text may be in English, Telugu or Hindi, so keep each language as it is. '
    'Mark unreadable parts as [illegible]. '
    'Output only the transcription, with no comments.'
)

def load_cache():
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
    return {}

def save_cache(cache):
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding='utf-8')

def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def split_table(md, max_chars=1500):
    lines = md.strip().split('\n')
    if len(md) <= max_chars or len(lines) < 4:
        return [md]
    header = lines[:2]
    parts, cur, size = [], [], 0
    for row in lines[2:]:
        if size + len(row) > max_chars and cur:
            parts.append('\n'.join(header + cur))
            cur, size = [], 0
        cur.append(row)
        size += len(row)
    if cur:
        parts.append('\n'.join(header + cur))
    return parts

def extract_tables(page):
    out = []
    try:
        for t in page.find_tables().tables:
            md = t.to_markdown()
            if md and len(md.strip()) > 20:
                out.extend(split_table(md))
    except Exception as e:
        print('  table extraction failed on page', page.number + 1, ':', e)
    return out

def is_scanned(page):
    # almost no text, but the page contains an image
    return len(page.get_text().strip()) < 50 and len(page.get_images()) > 0

def ocr_page(page, name, cache):
    key = name + '|' + str(page.number + 1)
    if key in cache:
        return cache[key]
    png = page.get_pixmap(dpi=150).tobytes('png')
    for attempt in range(3):
        try:
            text = ask_with_image(png, OCR_PROMPT) or ''
            break
        except Exception as e:
            print('  OCR retry', attempt + 1, 'on page', page.number + 1, ':', e)
            time.sleep(10 * (attempt + 1))
    else:
        return ''          # all retries failed; do not cache a failure
    cache[key] = text
    save_cache(cache)
    time.sleep(4)          # stay under the free-tier rate limit
    return text

def ingest_pdf(path, tables=True, ocr=True):
    doc = pymupdf.open(path)
    name = Path(path).name
    cache = load_cache() if ocr else {}
    ids, texts, metas = [], [], []
    counts = {'text': 0, 'table': 0, 'scan': 0}
    for page_no, page in enumerate(doc, start=1):
        pieces = []
        if ocr and is_scanned(page):
            print('  OCR on', name, 'page', page_no)
            ocr_text = ocr_page(page, name, cache).strip()
            if ocr_text:
                if len(ocr_text) <= 1500:
                    pieces.append((ocr_text, 'scan'))
                else:
                    for c in chunk_text(ocr_text, 1200, 150):
                        pieces.append((c, 'scan'))
        else:
            text = page.get_text().strip()
            if text:
                for c in chunk_text(text):
                    pieces.append((c, 'text'))
            if tables:
                for t in extract_tables(page):
                    pieces.append((t, 'table'))
        for j, (content, kind) in enumerate(pieces):
            ids.append(name + '-p' + str(page_no) + '-' + kind + str(j))
            texts.append(content)
            metas.append({'source': name, 'page': page_no, 'type': kind})
            counts[kind] += 1
    if texts:
        col.upsert(ids=ids, documents=texts, metadatas=metas,
                   embeddings=embed_passages(texts))
    print('Ingested', name, '-', counts['text'], 'text,', counts['table'], 'table,',
          counts['scan'], 'scan chunks')

if __name__ == '__main__':
    for pdf in Path('data/raw').glob('*.pdf'):
        ingest_pdf(str(pdf))