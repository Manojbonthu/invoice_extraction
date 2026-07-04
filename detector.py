"""Detect if a PDF is digital, scanned, or mixed (per-page).

WHAT'S NEW (in simple words):
  Just one change here — uses proper logging (logger) now instead of
  print(), so these messages get saved into the log file too, not just
  shown on screen and then lost.
"""

import logging
import fitz
from typing import Tuple, List

logger = logging.getLogger(__name__)


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
            logger.debug(f"Page {page_num + 1}: {page_type} ({text_len} chars)")
    finally:
        doc.close()

    if all(t == "digital" for t in per_page):
        overall = "digital"
    elif all(t == "scanned" for t in per_page):
        overall = "scanned"
    else:
        overall = "mixed"

    logger.info(f"{file_path}: overall PDF type = {overall} ({len(per_page)} page(s))")
    return overall, per_page