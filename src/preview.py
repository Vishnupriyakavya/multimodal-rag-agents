from pathlib import Path
import pymupdf

RAW = Path('data/raw')

def _highlight_text(page, text):
    # chunks come from get_text(), so each line exists on the page and can be searched
    n = 0
    for line in text.split('\n')[:60]:
        line = line.strip()
        if len(line) < 12:
            continue
        for rect in page.search_for(line[:120]):
            page.add_highlight_annot(rect)
            n += 1
    return n

def _mark_table(page, chunk):
    # find the table whose markdown contains this chunk's first data row, then outline it
    rows = chunk.strip().split('\n')
    probe = rows[2] if len(rows) > 2 else rows[0]
    try:
        for t in page.find_tables().tables:
            if probe in t.to_markdown():
                page.draw_rect(pymupdf.Rect(t.bbox), color=(1, 0, 0), width=2)
                return True
    except Exception:
        pass
    return False

def render_source(source, page_no, chunk='', kind='text', dpi=110):
    path = RAW / source
    if not path.exists():
        return None, False
    doc = pymupdf.open(path)
    page = doc[page_no - 1]
    found = False
    if kind == 'text':
        found = _highlight_text(page, chunk) > 0
    elif kind == 'table':
        found = _mark_table(page, chunk)
    png = page.get_pixmap(dpi=dpi).tobytes('png')
    doc.close()
    return png, found