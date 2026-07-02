"""
run.py - Full pipeline, per PDF. Both digital and scanned now produce the
SAME dict schema {"tables": [...], "texts": [...]} (+ "images" for scanned),
so both go through the same LLM pathway (get_invoice_json_from_data /
format_data_for_llm), which now also runs a Gemma vision fallback for any
hsn_sac/item_code still null after RuleEngine validation.

  DIGITAL:  digital_extractor.extract_digital_clean()  -> dict
  SCANNED:  scanned_extractor.extract_scanned_clean()   -> dict (+ images,
            already rendered for OCR, reused for the Gemma fallback)

Both paths:
  1. Detect type
  2. Extract -> dict
  3. Save raw extraction  -> Output_Folder/JSON_Raw/<name>.json (images excluded)
  4. Send dict to Gemini (format_data_for_llm), validate with RuleEngine,
     Gemma vision fallback if still null
  5. Save clean JSON      -> Output_Folder/JSON/<name>.json

Usage:
    python run.py                          # process every PDF in 1.Input
    python run.py "1.Input/invoice.pdf"    # process a single PDF
"""

import sys
import time
import json
from pathlib import Path

from config import INPUT_DIR
from detector import detect_pdf_type
from digital_extractor import extract_digital_clean
from scanned_extractor import extract_scanned_clean
from llm_extractor import get_invoice_json_from_data

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = BASE_DIR / "Output_Folder"
JSON_FOLDER = OUTPUT_FOLDER / "JSON"
JSON_RAW_FOLDER = OUTPUT_FOLDER / "JSON_Raw"


def collect_pdfs(root_dir: str):
    root = Path(root_dir)
    pdfs = []
    for path in root.rglob("*.pdf"):
        if "processed" in path.parts or "Job Work" in path.parts:
            continue
        if "1.Input" in path.parts:
            pdfs.append(path)
    return sorted(pdfs)


def process_pdf(pdf_path: Path) -> bool:
    fname = pdf_path.name
    doc_id = pdf_path.stem
    print(f"\nProcessing: {fname}")

    # 1. Detect type
    overall, _ = detect_pdf_type(str(pdf_path))
    print(f"Type: {overall}")

    # 2. Extract (dict schema, same for both paths)
    try:
        if overall == "digital":
            data = extract_digital_clean(str(pdf_path))
        else:
            data = extract_scanned_clean(str(pdf_path))
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

    if not data.get("tables") and not data.get("texts"):
        print(f"No data extracted for {fname}")
        return False

    print(f"Extracted: {len(data.get('tables', []))} tables, {len(data.get('texts', []))} text blocks")

    # 3. Save raw extraction (exclude "images" — PIL images aren't JSON-serializable)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    JSON_RAW_FOLDER.mkdir(parents=True, exist_ok=True)
    raw_path = JSON_RAW_FOLDER / f"{doc_id}.json"
    raw_for_disk = {k: v for k, v in data.items() if k != "images"}
    raw_path.write_text(json.dumps(raw_for_disk, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved RAW JSON: {raw_path}")

    # 4. Send dict to Gemini (+ Gemma vision fallback if needed)
    try:
        result = get_invoice_json_from_data(data, pdf_path=str(pdf_path))
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return False

    # 5. Save clean JSON
    JSON_FOLDER.mkdir(parents=True, exist_ok=True)
    json_path = JSON_FOLDER / f"{doc_id}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved JSON: {json_path}")

    return True


def main():
    print("=" * 60)
    print("FULL PIPELINE (Digital + Scanned) -> LLM -> Gemma Vision Fallback -> JSON")
    print("=" * 60)

    if len(sys.argv) > 1:
        pdfs = [Path(sys.argv[1])]
        if not pdfs[0].exists():
            print(f"File not found: {pdfs[0]}")
            return
    else:
        pdfs = collect_pdfs(INPUT_DIR)
        if not pdfs:
            print(f"No PDFs found in {INPUT_DIR} (looking for 1.Input folders).")
            return

    total_start = time.time()
    success = 0
    failed = 0

    for pdf in pdfs:
        if process_pdf(pdf):
            success += 1
        else:
            failed += 1

    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"SUMMARY: Success: {success}, Failed: {failed}")
    print(f"Total time: {total_time:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()