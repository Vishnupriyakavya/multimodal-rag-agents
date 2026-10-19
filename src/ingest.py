from pathlib import Path
import pymupdf
import chromadb
from src.embed import embed_passages

db = chromadb.PersistentClient(path='db')
col = db.get_or_create_collection('docs', metadata={'hnsw:space': 'cosine'})

def chunk_text(text, size=800, overlap=150):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def split_table(md, max_chars=1500):
    # long tables are split by rows, repeating the header in every part
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

def ingest_pdf(path, tables=True):
    doc = pymupdf.open(path)
    name = Path(path).name
    ids, texts, metas = [], [], []
    counts = {'text': 0, 'table': 0}
    for page_no, page in enumerate(doc, start=1):
        pieces = []
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
    print('Ingested', name, '-', counts['text'], 'text chunks,', counts['table'], 'table chunks')

if __name__ == '__main__':
    for pdf in Path('data/raw').glob('*.pdf'):
        ingest_pdf(str(pdf))