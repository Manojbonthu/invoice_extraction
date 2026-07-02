"""Detect if a PDF is digital, scanned, or mixed (per-page)."""

import fitz
from typing import Tuple, List


def detect_pdf_type(file_path: str) -> Tuple[str, List[str]]:
    doc = fitz.open(file_path)
    per_page: List[str] = []
    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().strip()
            text_len = len(text)
            page_type = "digital" if text_len > 5 else "scanned"
            per_page.append(page_type)
            print(f"Page {page_num+1}: {page_type} ({text_len} chars)")
    finally:
        doc.close()

    if all(t == "digital" for t in per_page):
        overall = "digital"
    elif all(t == "scanned" for t in per_page):
        overall = "scanned"
    else:
        overall = "mixed"

    print(f"\nOverall PDF type: {overall}")
    return overall, per_page