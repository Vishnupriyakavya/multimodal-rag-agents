import pymupdf

src = pymupdf.open('data/raw/PMKisanSamanNidhi.pdf')
out = pymupdf.open()
for i in range(3):
    pix = src[i].get_pixmap(dpi=110)
    page = out.new_page(width=src[i].rect.width, height=src[i].rect.height)
    page.insert_image(page.rect, stream=pix.tobytes('png'))
out.save('data/raw/test_scan.pdf')
print('saved test_scan.pdf')