"""
test_pdf.py – Pass one PDF, auto-detect digital/scanned, save plain text
to Output_Folder. Confirms the whole detect -> extract chain is in sync.

Usage:
    python test_pdf.py "Digital_PDF/1.pdf"
    python test_pdf.py "Scanned_PDF/1.pdf"
"""

import sys
from pathlib import Path

from detector import detect_pdf_type
from digital_extractor import extract_digital_text_only
from scanned_extractor import extract_scanned_text_only

OUTPUT_DIR = Path(__file__).parent / "Output_Folder"


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_pdf.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Checking: {pdf_path}")
    overall, per_page = detect_pdf_type(str(pdf_path))
    print(f"Detected type: {overall}")

    if overall == "digital":
        print("\nRunning digital extraction (PyMuPDF)...")
        text = extract_digital_text_only(str(pdf_path))
    else:
        # "scanned" or "mixed" -> OCR path
        print("\nRunning OCR extraction (Surya OCR, may take a moment)...")
        text = extract_scanned_text_only(str(pdf_path))

    if not text.strip():
        print("⚠️  No text extracted — check the PDF or extractor.")
        sys.exit(1)

    out_path = OUTPUT_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Saved: {out_path}")
    print(f"Characters extracted: {len(text)}")
    print("=" * 60)


if __name__ == "__main__":
    main()