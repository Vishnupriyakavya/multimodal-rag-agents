from pathlib import Path
import pymupdf
from src.ingest import has_visual, is_scanned

for pdf in sorted(Path('data/raw').glob('*.pdf')):
    doc = pymupdf.open(pdf)
    n = sum(1 for p in doc if not is_scanned(p) and has_visual(p))
    print(pdf.name, '| pages:', len(doc), '| candidate pages for captioning:', n)