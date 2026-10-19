import difflib
import json
import pymupdf

src = pymupdf.open('data/raw/pmkisan_guidelines.pdf')
cache = json.load(open('cache/ocr.json', encoding='utf-8'))

for i in range(3):
    truth = src[i].get_text().split()
    got = cache.get('test_scan.pdf|' + str(i + 1), '').split()
    ratio = difflib.SequenceMatcher(None, truth, got).ratio()
    print('page', i + 1, '| word-level similarity:', round(ratio, 3))