"""
test_from_raw_json.py - Send already-extracted raw JSON (Output_Folder/JSON_Raw/*.json)
straight to the LLM, skipping detection/extraction entirely. If hsn_sac or item_code
is still null after RuleEngine validation, the Gemma vision fallback kicks in
automatically (already built into get_invoice_json_from_data) - it just needs the
original PDF to render a page image from, so this script locates that PDF by name.

Usage:
    python test_from_raw_json.py 1_digital          # one file (no .json needed)
    python test_from_raw_json.py 1_digital.json      # .json extension also fine
    python test_from_raw_json.py                     # process every file in JSON_Raw
"""

import sys
import json
from pathlib import Path
from typing import Optional

from llm_extractor import get_invoice_json_from_data

BASE_DIR = Path(__file__).parent
JSON_RAW_FOLDER = BASE_DIR / "Output_Folder" / "JSON_Raw"
JSON_FOLDER = BASE_DIR / "Output_Folder" / "JSON"

# Folders to search for the matching source PDF (add more if your layout differs)
PDF_SEARCH_DIRS = ["1.Input", "Digital_PDF", "Scanned_PDF", "PDF"]


def find_matching_pdf(doc_id: str) -> Optional[Path]:
    """Find the original PDF matching this doc_id by filename stem."""
    for folder_name in PDF_SEARCH_DIRS:
        folder = BASE_DIR / folder_name
        if not folder.exists():
            continue
        for pdf_path in folder.rglob("*.pdf"):
            if pdf_path.stem == doc_id:
                return pdf_path
    return None


def process_raw_json(raw_json_path: Path) -> bool:
    doc_id = raw_json_path.stem
    print(f"\nProcessing: {raw_json_path.name}")

    try:
        data = json.loads(raw_json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Failed to read raw JSON: {e}")
        return False

    if not data.get("tables") and not data.get("texts"):
        print(f"Raw JSON has no tables/texts for {doc_id}")
        return False

    pdf_path = find_matching_pdf(doc_id)
    if pdf_path:
        print(f"Matched PDF: {pdf_path}")
    else:
        print(f"No matching PDF found for '{doc_id}' - Gemma vision fallback "
              f"will be skipped if any field is still null (searched: {PDF_SEARCH_DIRS})")

    try:
        result = get_invoice_json_from_data(data, pdf_path=str(pdf_path) if pdf_path else None)
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return False

    JSON_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = JSON_FOLDER / f"{doc_id}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_path}")

    return True


def main():
    if not JSON_RAW_FOLDER.exists():
        print(f"JSON_Raw folder not found: {JSON_RAW_FOLDER}")
        return

    if len(sys.argv) > 1:
        name = sys.argv[1]
        if not name.endswith(".json"):
            name += ".json"
        raw_files = [JSON_RAW_FOLDER / name]
        if not raw_files[0].exists():
            print(f"File not found: {raw_files[0]}")
            return
    else:
        raw_files = sorted(JSON_RAW_FOLDER.glob("*.json"))
        if not raw_files:
            print(f"No JSON files found in {JSON_RAW_FOLDER}")
            return

    success = 0
    failed = 0

    for raw_path in raw_files:
        if process_raw_json(raw_path):
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: Success: {success}, Failed: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()