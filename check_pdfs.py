from pathlib import Path
import pymupdf

for pdf in Path('data/raw').glob('*.pdf'):
    doc = pymupdf.open(pdf)
    chars = sum(len(p.get_text()) for p in doc)
    print(pdf.name, '| pages:', len(doc), '| characters:', chars)