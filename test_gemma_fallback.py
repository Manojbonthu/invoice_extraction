"""
test_gemma_fallback.py – Standalone test: skip extraction, load an already-saved
raw JSON (tables/texts from Output_Folder/JSON_Raw/), send it to Gemini,
and if hsn_sac/item_code is still null after RuleEngine, run the Gemma
vision fallback using the page image rendered from the original PDF.

Usage:
    python test_gemma_fallback.py "Output_Folder/JSON_Raw/2_Scanned.json" "PDF/2_Scanned.pdf"
"""

import sys
import json
from pathlib import Path

from detector import detect_pdf_type
from digital_extractor import render_page_images
from scanned_extractor import pdf_to_images
from llm_extractor import (
    format_data_for_llm,
    call_api,
    parse_response,
    apply_gemma_vision_fallback,
)
from rule_engine import RuleEngine


def count_nulls(invoices):
    n = 0
    for inv in invoices:
        for item in inv.get("Invoice Items", []):
            if item.get("hsn_sac") is None:
                n += 1
            if item.get("item_code") is None:
                n += 1
    return n


def main():
    if len(sys.argv) < 3:
        print("Usage: python test_gemma_fallback.py <raw_json_path> <original_pdf_path>")
        sys.exit(1)

    raw_json_path = Path(sys.argv[1])
    pdf_path = Path(sys.argv[2])

    if not raw_json_path.exists():
        print(f"ERROR: JSON not found: {raw_json_path}")
        sys.exit(1)
    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}")
        sys.exit(1)

    # 1. Load the already-extracted raw data (tables/texts) — NO extraction re-run
    print(f"Loading raw JSON: {raw_json_path}")
    with open(raw_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("tables") and not data.get("texts"):
        print("ERROR: Raw JSON has no tables/texts — nothing to send to LLM.")
        sys.exit(1)

    # 2. Send to Gemini
    print("\nSending to Gemini...")
    prompt = format_data_for_llm(data)
    response = call_api(prompt)
    extracted, in_tok, out_tok, tot_tok = parse_response(response)
    print(f"Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")

    was_list = isinstance(extracted, list)
    invoices = extracted if was_list else [extracted]

    # 3. RuleEngine validation
    invoices = RuleEngine.apply_post_llm_rules(invoices)

    nulls_before = count_nulls(invoices)
    print(f"\nNull fields after Gemini + RuleEngine: {nulls_before}")
    print(json.dumps(invoices, indent=2, ensure_ascii=False))

    # 4. Gemma vision fallback, only if nulls remain
    if nulls_before > 0:
        print(f"\n🔍 {nulls_before} field(s) still null — running Gemma vision fallback")

        # Render page image(s) from the original PDF — no full re-extraction
        overall, _ = detect_pdf_type(str(pdf_path))
        if overall == "digital":
            images = render_page_images(str(pdf_path))
        else:
            images = pdf_to_images(str(pdf_path))

        invoices = apply_gemma_vision_fallback(invoices, images)

        nulls_after = count_nulls(invoices)
        print(f"\nNull fields after Gemma vision fallback: {nulls_after}")
        print(f"Fixed: {nulls_before - nulls_after} field(s)")
    else:
        print("\nNo nulls — Gemma vision fallback not triggered.")

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    final = invoices if was_list else invoices[0]
    print(json.dumps(final, indent=2, ensure_ascii=False))

    # Save to a separate test output so it doesn't clobber your real pipeline output
    out_path = raw_json_path.parent.parent / "JSON_TestFallback" / raw_json_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved test result: {out_path}")


if __name__ == "__main__":
    main()