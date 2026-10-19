from pathlib import Path
import pymupdf  # PyMuPDF
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

def ingest_pdf(path):
    doc = pymupdf.open(path)
    name = Path(path).name
    ids, texts, metas = [], [], []
    for page_no, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if not text:
            continue
        for j, chunk in enumerate(chunk_text(text)):
            ids.append(name + '-p' + str(page_no) + '-c' + str(j))
            texts.append(chunk)
            metas.append({'source': name, 'page': page_no, 'type': 'text'})
    if texts:
        col.upsert(ids=ids, documents=texts, metadatas=metas,
                   embeddings=embed_passages(texts))
    print('Ingested', name, '-', len(texts), 'chunks')

if __name__ == '__main__':
    for pdf in Path('data/raw').glob('*.pdf'):
        ingest_pdf(str(pdf))