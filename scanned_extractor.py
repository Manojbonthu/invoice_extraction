"""
scanned_extractor.py - Reads text out of SCANNED (image-based) PDF invoices
using Surya OCR. OCR means "Optical Character Recognition" — basically,
turning a picture of text into real, computer-readable text.

WHAT'S NEW (in simple words):
  1. Uses proper logging now (logger) instead of print().
  2. Reads settings from config.py instead of hardcoding numbers. This
     means the SAME code automatically behaves correctly on your CPU
     laptop and on your company's GPU server — no code changes needed,
     just different settings in the .env file.
  3. Loads the OCR model onto the correct device (CPU or GPU) based on
     config.DEVICE, instead of letting the library guess by itself.
  4. Processes pages in BATCHES (many pages sent to OCR at once) instead
     of one page at a time in a loop. On a GPU this is much faster. The
     batch size is controlled by config.OCR_BATCH_SIZE (small on CPU,
     bigger on GPU).
  5. Each text block now also stores which PAGE it came from. This is
     needed later for multi-page invoices, so we know which page to show
     to the Gemma vision fallback if a field is missing.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import pypdfium2 as pdfium
from PIL import Image

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

import config

logger = logging.getLogger(__name__)

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


# --- Global model cache (load once per process, reuse for every PDF) ------
# IMPORTANT: on CPU, keep OCR_WORKERS=1 in your .env — loading this model
# more than once at the same time just fights over the same CPU cores and
# uses more RAM, with no speed benefit. On GPU, a couple of workers can
# make sense if you have enough VRAM (GPU memory) free.
_predictors = None


def _get_predictors():
    """
    Loads the Surya OCR models one time, onto the correct device (CPU or
    GPU), and reuses them for every PDF after that. Loading a model is
    slow, so we only want to do it once, not per file.
    """
    global _predictors
    if _predictors is None:
        logger.info(f"Loading Surya OCR models onto device: {config.DEVICE} "
                    f"(this happens once per run)")
        foundation_predictor = FoundationPredictor(device=config.DEVICE)
        recognition_predictor = RecognitionPredictor(foundation_predictor)
        detection_predictor = DetectionPredictor(device=config.DEVICE)
        _predictors = (recognition_predictor, detection_predictor)
        logger.info("Surya OCR models loaded successfully.")
    return _predictors


def pdf_to_images(pdf_path: str, dpi: int = None) -> List[Image.Image]:
    """
    Turns each page of the PDF into a picture (image), so OCR can "read"
    it like a photo. Higher dpi = sharper image = better accuracy, but
    slower and uses more memory. Defaults to config.OCR_DPI if not given
    (150 on CPU, 200 on GPU, unless you set OCR_DPI yourself in .env).
    """
    dpi = dpi or config.OCR_DPI
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


def _batched(items: list, batch_size: int):
    """Splits a list into smaller chunks of a given size.
    Example: _batched([1,2,3,4,5], 2) -> [1,2], [3,4], [5]
    This is how we send pages to OCR in small groups instead of all at
    once (which could run out of memory) or one at a time (which is slow
    on GPU)."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def extract_scanned_clean(pdf_path: str, dpi: int = None) -> Dict[str, Any]:
    """
    Extract plain OCR text from a scanned PDF.

    Returns: {
        "tables": [],
        "texts": [{"type": "text"/"heading", "text": "...", "page": 1}, ...],
        "images": [PIL.Image, ...]   # one image per page, same order
    }

    The "page" number (starting at 1) is now included on every text block,
    so later steps (like the Gemma vision fallback) know exactly which
    page image to look at if a field is still missing.

    The images are the same page pictures already used for OCR — we keep
    them here instead of throwing them away, so we don't have to redo the
    PDF-to-image conversion again later.
    """
    recognition_predictor, detection_predictor = _get_predictors()
    images = pdf_to_images(pdf_path, dpi=dpi)

    result: Dict[str, Any] = {"tables": [], "texts": [], "images": images}

    if not images:
        logger.warning(f"No pages rendered for {pdf_path} — nothing to OCR.")
        return result

    batch_size = max(1, config.OCR_BATCH_SIZE)
    page_number = 1  # page numbers start at 1, not 0, to match how humans count pages

    for image_batch in _batched(images, batch_size):
        logger.info(f"Running OCR on {len(image_batch)} page(s) "
                    f"(batch size {batch_size}, device {config.DEVICE})")
        page_preds = recognition_predictor(image_batch, det_predictor=detection_predictor)

        for page_pred in page_preds:
            for para in _lines_to_paragraphs(page_pred):
                is_heading = para.isupper() and len(para) < 100
                result["texts"].append({
                    "type": "heading" if is_heading else "text",
                    "text": para,
                    "page": page_number,
                })
            page_number += 1

    logger.info(f"OCR finished for {pdf_path}: {len(result['texts'])} text blocks "
                f"across {len(images)} page(s)")
    return result


def extract_scanned_text_only(pdf_path: str, dpi: int = None) -> str:
    """
    Convenience function: return scanned PDF as a single plain-text string
    (no dict wrapper). This is what gets sent to the LLM in text-only mode.
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