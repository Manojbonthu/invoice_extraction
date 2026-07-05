"""
digital_extractor.py - Reads tables and text directly out of DIGITAL PDFs
(PDFs that already have real, selectable text in them — not scanned
pictures). This is much faster and more accurate than OCR, when it works.

WHAT'S NEW (in simple words):
  1. Uses proper logging now (logger) instead of print().
  2. Reads DPI (image sharpness) from config.py instead of a hardcoded
     number, so it matches whatever your CPU/GPU settings say.
  3. Every text block and table now remembers which PAGE it came from
     (just like scanned_extractor.py now does). This is needed for
     multi-page invoices, so the Gemma vision fallback in
     llm_extractor.py knows exactly which page image to check.
  4. render_page_images() now also uses config.OCR_DPI as its default,
     so digital and scanned PDFs render pages at a consistent,
     device-appropriate resolution.

FIXED: extract_digital_clean() was calling len(doc) in its final log
line AFTER doc.close() had already run in the `finally` block. PyMuPDF
raises RuntimeError("document closed") the instant you touch a closed
Document — that's what was crashing every digital PDF instantly, before
it ever reached the LLM step. Page count is now captured up front while
the doc is still open.
"""

import os
import re
import io
import logging
import fitz
from typing import List, Dict, Any
from PIL import Image

import config

logger = logging.getLogger(__name__)

# --- Block character filter ------------------------------------------------
BLOCK_CHARS_RE = re.compile(r'[█▀▄▌▐░▒▓⎔▬►▼◄◆◇●○■□]')


def clean_span_text(text: str) -> str:
    """Remove block characters and collapse whitespace."""
    if not text:
        return ""
    cleaned = BLOCK_CHARS_RE.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


# --- Lazy page-image rendering for the Gemma vision fallback ---------------
def render_page_images(pdf_path: str, dpi: int = None) -> List[Image.Image]:
    """
    Turns each page of a digital PDF into a picture (image). Only called
    on-demand from llm_extractor.py when a field is still missing after
    the Gemini text pass — most digital PDFs never need this, since the
    text pass usually finds everything.

    dpi (sharpness) defaults to config.OCR_DPI if not given, so digital
    and scanned PDFs use a consistent, device-appropriate resolution
    (lower on CPU to save memory/time, higher on GPU).
    """
    dpi = dpi or config.OCR_DPI
    doc = fitz.open(pdf_path)
    images = []
    try:
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            images.append(Image.open(io.BytesIO(img_bytes)))
    finally:
        doc.close()
    logger.info(f"Rendered {len(images)} page image(s) from {pdf_path} at {dpi} DPI "
                f"for the Gemma vision fallback")
    return images


# --- PHASE 1: table detection (math-based, no keywords) --------------------
def _is_valid_data_table(headers, rows):
    """
    Accept any table with reasonable structure - size + density checks only.
    LLM filters out irrelevant tables downstream.
    """
    if not headers or not rows:
        return False
    if len(headers) < 3:
        return False
    if len(rows) < 3:
        return False

    total_cells = len(headers) + (len(rows) * len(headers))
    filled_cells = 0

    for h in headers:
        if h and len(str(h).strip()) > 0:
            filled_cells += 1

    for row in rows:
        for cell in row:
            if cell and len(str(cell).strip()) > 0:
                filled_cells += 1

    density = filled_cells / total_cells if total_cells > 0 else 0
    return density > 0.20  # at least 20% filled


# --- PHASE 2: extract ALL text blocks (including overlaps) -----------------
def _extract_all_text_blocks(page, table_bboxes: List = None):
    """
    Extract ALL text from page without skipping table overlaps.
    Captures metadata, headers, and footer text that might overlap tables.
    LLM handles deduplication downstream.

    Returns: List of {"type": "text"/"heading", "text": "...", "bbox": [...]}
    """
    if table_bboxes is None:
        table_bboxes = []

    text_blocks = []
    text_dict = page.get_text("dict")

    for block in text_dict.get("blocks", []):
        if block["type"] != 0:  # skip non-text blocks (images, etc)
            continue

        for line in block.get("lines", []):
            spans = []
            line_bbox = line.get("bbox", [])

            for span in line.get("spans", []):
                txt = clean_span_text(span["text"])
                if txt:
                    spans.append(txt)

            if spans:
                line_text = " ".join(spans)
                is_heading = (
                    line_text.isupper()
                    and len(line_text) < 80
                    and len(line_text.split()) <= 5
                )
                text_blocks.append({
                    "type": "heading" if is_heading else "text",
                    "text": line_text,
                    "bbox": line_bbox,
                })

    return text_blocks


# --- MAIN EXTRACTION ----------------------------------------------------
def extract_digital_clean(pdf_path: str) -> Dict[str, Any]:
    """
    Extract clean tables and text from a digital PDF.

    Returns: {
        "tables": [{"headers": [...], "rows": [[...]], "page": 1}, ...],
        "texts": [{"type": "text"/"heading", "text": "...", "page": 1}, ...]
    }

    Every table and text block now includes a "page" number (starting at
    1), matching what scanned_extractor.py does — needed so the Gemma
    vision fallback in llm_extractor.py knows which page image to check
    if a field turns out to be missing.

    Note: no "images" key here (unlike scanned) — call render_page_images()
    separately, lazily, only if the Gemma vision fallback is needed.
    """
    doc = fitz.open(pdf_path)
    result = {"tables": [], "texts": []}
    page_count = len(doc)  # captured while doc is still open — used in the log line below

    try:
        for page_index, page in enumerate(doc, start=1):
            table_bboxes = []

            # Table extraction (math-based filtering)
            try:
                tables = page.find_tables()
                for table in tables.tables:
                    table_bboxes.append(table.bbox)
                    extracted = table.extract()

                    if not extracted:
                        continue

                    headers = []
                    if table.header:
                        headers = [clean_span_text(str(h)) if h else "" for h in table.header.names]

                    rows = []
                    for row in extracted:
                        clean_row = [clean_span_text(str(c)) if c else "" for c in row]
                        if any(cell for cell in clean_row):
                            rows.append(clean_row)

                    if _is_valid_data_table(headers, rows):
                        result["tables"].append({
                            "headers": headers,
                            "rows": rows,
                            "page": page_index,
                        })
                    else:
                        if headers or rows:
                            logger.debug(f"Skipped small/sparse table on page {page_index} "
                                         f"(cols:{len(headers)}, rows:{len(rows)})")

            except Exception as e:
                logger.warning(f"Table extraction error on page {page_index} of {pdf_path}: {e}")
                continue

            # Extract ALL text (no overlap skipping)
            try:
                text_blocks = _extract_all_text_blocks(page, table_bboxes)
                for block in text_blocks:
                    result["texts"].append({
                        "type": block["type"],
                        "text": block["text"],
                        "page": page_index,
                    })
            except Exception as e:
                logger.warning(f"Text extraction error on page {page_index} of {pdf_path}: {e}")
                continue

    finally:
        doc.close()

    logger.info(f"Digital extraction finished for {pdf_path}: "
                f"{len(result['tables'])} table(s), {len(result['texts'])} text block(s) "
                f"across {page_count} page(s)")
    return result


# --- LEGACY WRAPPER (for compatibility) -------------------------------------
from schemas import NormalizedBlock, SourceRef
import uuid


def extract_digital(pdf_path: str, document_id: str = "default") -> List[NormalizedBlock]:
    data = extract_digital_clean(pdf_path)
    blocks = []
    filename = os.path.basename(pdf_path)

    for table in data["tables"]:
        headers = table["headers"]
        rows = table["rows"]

        header_row = "| " + " | ".join(headers) + " |" if headers else ""
        separator = "| " + " | ".join(["---"] * len(headers)) + " |" if headers else ""
        data_rows = ["| " + " | ".join(r) + " |" for r in rows]
        md_parts = [p for p in [header_row, separator] + data_rows if p]

        blocks.append(NormalizedBlock(
            block_id=str(uuid.uuid4()),
            document_id=document_id,
            type="table",
            text="\n".join(md_parts),
            table_data={"headers": headers, "rows": rows},
            source_ref=SourceRef(filename=filename, page=table.get("page", 1)),
            confidence=0.95,
        ))

    for text in data["texts"]:
        blocks.append(NormalizedBlock(
            block_id=str(uuid.uuid4()),
            document_id=document_id,
            type=text["type"],
            text=text["text"],
            source_ref=SourceRef(filename=filename, page=text.get("page", 1)),
            confidence=1.0,
        ))

    return blocks


if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python digital_extractor.py <pdf_path> [output.json]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(pdf_path).with_suffix(".json")

    data = extract_digital_clean(pdf_path)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON: {out_path}")