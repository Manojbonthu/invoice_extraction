"""
digital_extractor.py - Extract tables and text from digital PDFs.
Removes block/box-drawing characters (block glyphs).
  - Phase 1: Math-based table detection (no hardcoded keywords)
  - Phase 2: Extract ALL text (no overlap skipping)

Returns dict: {"tables": [{"headers": [...], "rows": [[...]]}, ...], "texts": [...]}
This is the JSON/table format that gets sent to the LLM via
llm_extractor.format_data_for_llm() / get_invoice_json_from_data().
"""

import os
import re
import fitz
from typing import List, Dict, Any

# --- Block character filter ------------------------------------------------
BLOCK_CHARS_RE = re.compile(r'[█▀▄▌▐░▒▓⎔▬►▼◄◆◇●○■□]')


def clean_span_text(text: str) -> str:
    """Remove block characters and collapse whitespace."""
    if not text:
        return ""
    cleaned = BLOCK_CHARS_RE.sub(' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()


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
    Returns: {"tables": [{"headers": [...], "rows": [[...]]}, ...], "texts": [...]}
    """
    doc = fitz.open(pdf_path)
    result = {"tables": [], "texts": []}

    try:
        for page in doc:
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
                        })
                    else:
                        if headers or rows:
                            print(f"  Skipped small/sparse table (cols:{len(headers)}, rows:{len(rows)})")

            except Exception as e:
                print(f"  Table extraction error: {e}")
                continue

            # Extract ALL text (no overlap skipping)
            try:
                text_blocks = _extract_all_text_blocks(page, table_bboxes)
                for block in text_blocks:
                    result["texts"].append({
                        "type": block["type"],
                        "text": block["text"],
                    })
            except Exception as e:
                print(f"  Text extraction error: {e}")
                continue

    finally:
        doc.close()

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
            source_ref=SourceRef(filename=filename, page=1),
            confidence=0.95,
        ))

    for text in data["texts"]:
        blocks.append(NormalizedBlock(
            block_id=str(uuid.uuid4()),
            document_id=document_id,
            type=text["type"],
            text=text["text"],
            source_ref=SourceRef(filename=filename, page=1),
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