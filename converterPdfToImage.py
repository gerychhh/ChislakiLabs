import fitz  # PyMuPDF

pdf_path = "input.pdf"
doc = fitz.open(pdf_path)

for page_index in range(len(doc)):
    page = doc[page_index]
    # dpi = 300 или 600 для качества
    zoom = 600 / 72  # 72 dpi базовое, умножаем для нужного dpi
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    filename = f"page_{page_index+1}.jpg"
    pix.save(filename)
    print(f"Сохранено: {filename}")
