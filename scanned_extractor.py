"""
scanned_extractor.py - Extract plain OCR text from scanned PDFs using Surya OCR 0.17.0.
Returns flat text only (no table reconstruction). Table/column ambiguity is
handled downstream by a dedicated, more detailed LLM prompt instead
(see llm_extractor.SCANNED_SYSTEM_PROMPT) - which turned out to be more
reliable than trying to geometrically reconstruct tables from line-level
OCR bboxes.
"""

import re
from pathlib import Path
from typing import List, Dict, Any

import pypdfium2 as pdfium
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# --- Text cleanup ---------------------------------------------------------
BLOCK_CHARS_RE = re.compile(r'[█▀▄▌▐░▒▓⎔▬►▼◄◆◇●○■□]')
TAG_RE = re.compile(r"</?b>")


def clean_text(text: str) -> str:
    if not text:
        return ""
    cleaned = TAG_RE.sub("", text)
    cleaned = BLOCK_CHARS_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


# --- Global model cache (load once, reuse across every PDF in a run) ------
_predictors = None


def _get_predictors():
    global _predictors
    if _predictors is None:
        foundation_predictor = FoundationPredictor()
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor()
        _predictors = (recognition_predictor, detection_predictor)
    return _predictors


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    pdf = pdfium.PdfDocument(pdf_path)
    scale = dpi / 72
    images = []
    for page in pdf:
        bitmap = page.render(scale=scale)
        images.append(bitmap.to_pil())
    return images


def _lines_to_paragraphs(page_pred) -> List[str]:
    """Sort lines top-to-bottom/left-to-right, group into paragraphs by vertical gap."""
    lines = []
    for line in page_pred.text_lines:
        x1, y1, x2, y2 = line.bbox
        txt = clean_text(line.text)
        if txt:
            lines.append({"text": txt, "bbox": (x1, y1, x2, y2)})

    if not lines:
        return []

    lines.sort(key=lambda l: (round(l["bbox"][1] / 10), l["bbox"][0]))

    paragraphs = []
    current = []
    prev = None
    for l in lines:
        if prev is None:
            current.append(l["text"])
        else:
            gap = l["bbox"][1] - prev["bbox"][3]
            height = max(prev["bbox"][3] - prev["bbox"][1], 1)
            if gap < 1.5 * height:
                current.append(l["text"])
            else:
                paragraphs.append(" ".join(current))
                current = [l["text"]]
        prev = l
    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def extract_scanned_clean(pdf_path: str, dpi: int = 150) -> Dict[str, Any]:
    """
    Extract plain OCR text from a scanned PDF.
    Returns: {"tables": [], "texts": [{"type": "text"/"heading", "text": "..."}]}
    """
    recognition_predictor, detection_predictor = _get_predictors()
    images = pdf_to_images(pdf_path, dpi=dpi)

    result: Dict[str, Any] = {"tables": [], "texts": []}

    for img in images:
        page_preds = recognition_predictor([img], det_predictor=detection_predictor)
        for page_pred in page_preds:
            for para in _lines_to_paragraphs(page_pred):
                is_heading = para.isupper() and len(para) < 100
                result["texts"].append({
                    "type": "heading" if is_heading else "text",
                    "text": para,
                })

    return result


def extract_scanned_text_only(pdf_path: str, dpi: int = 150) -> str:
    """
    Convenience function: return scanned PDF as a single plain-text string
    (no dict wrapper). This is what gets sent to the LLM.
    """
    data = extract_scanned_clean(pdf_path, dpi=dpi)
    return "\n".join(t["text"] for t in data["texts"])


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scanned_extractor.py <pdf_path> [output.txt]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(pdf_path).with_suffix(".txt")

    text = extract_scanned_text_only(pdf_path)
    out_path.write_text(text, encoding="utf-8")
    print(f"Saved TEXT: {out_path}")