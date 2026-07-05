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

  NEW — CROSS-PDF BATCHING (create_pdf_aware_batches + extract_scanned_batch):
     Before this, if you had 10 scanned PDFs and several worker threads,
     each thread called the OCR model separately for its own PDF — many
     threads fighting over the same GPU model at once. That risks GPU
     out-of-memory errors and wastes GPU time on lots of small calls
     instead of one efficient big one.

     Now, run.py can gather up several scanned PDFs, group them by total
     PAGE COUNT (never splitting one PDF's pages across two groups), and
     send ALL of a group's pages through the OCR model in ONE call. This
     keeps GPU usage to one big, orderly batch at a time instead of many
     threads competing for the same model.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _count_pages(pdf_path: str) -> int:
    """How many pages does this PDF have? Used to build page-budget
    batches without ever having to fully render a PDF just to count it."""
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        return len(pdf)
    except Exception as e:
        logger.error(f"Could not count pages for {pdf_path}: {e}")
        return 0


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
    Extract plain OCR text from ONE scanned PDF (single-file version —
    still used by test scripts / one-off runs). For processing MANY
    scanned PDFs efficiently on a GPU, see create_pdf_aware_batches() +
    extract_scanned_batch() below instead.

    Returns: {
        "tables": [],
        "texts": [{"type": "text"/"heading", "text": "...", "page": 1}, ...],
        "images": [PIL.Image, ...]   # one image per page, same order
    }
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


# ─────────────────────────────────────────────────────────
# NEW: CROSS-PDF PAGE-BUDGET BATCHING
# Groups several scanned PDFs together so one OCR call can handle all of
# them at once, instead of one call per PDF fighting over the same GPU.
# ─────────────────────────────────────────────────────────

def create_pdf_aware_batches(pdf_paths: List[str], batch_size: int = None) -> List[List[str]]:
    """
    Groups PDFs into batches where the total PAGE COUNT per batch stays
    within batch_size, WITHOUT ever splitting one PDF's pages across two
    batches.

    Example with batch_size=10:
      PDF A (4 pages), PDF B (3 pages), PDF C (5 pages), PDF D (2 pages)
      -> Batch 1: [A, B]   (4+3=7 pages, adding C would make 12 > 10)
      -> Batch 2: [C, D]   (5+2=7 pages)

    A single PDF bigger than batch_size gets its own batch by itself —
    it's never split.
    """
    batch_size = batch_size or config.OCR_BATCH_SIZE
    batches: List[List[str]] = []
    current_batch: List[str] = []
    current_pages = 0

    for pdf_path in pdf_paths:
        count = _count_pages(pdf_path)
        if count == 0:
            logger.warning(f"Skipping unreadable/empty PDF for OCR batching: {pdf_path}")
            continue

        if current_batch and (current_pages + count > batch_size):
            batches.append(current_batch)
            current_batch = []
            current_pages = 0

        current_batch.append(pdf_path)
        current_pages += count

    if current_batch:
        batches.append(current_batch)

    logger.info(f"Grouped {len(pdf_paths)} scanned PDF(s) into {len(batches)} OCR batch(es) "
                f"(page budget per batch: {batch_size})")
    return batches


def extract_scanned_batch(pdf_paths: List[str], dpi: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Runs OCR on MULTIPLE scanned PDFs together, in ONE call to the model,
    instead of one call per PDF. This is the GPU-efficient version of
    extract_scanned_clean() — call create_pdf_aware_batches() first to
    build a sensible group of PDFs to pass in here (one whose total page
    count fits your OCR_BATCH_SIZE).

    Returns: { pdf_path: {"tables": [], "texts": [...], "images": [...]}, ... }
    — same per-file shape as extract_scanned_clean() has always returned,
    just for many files at once, so the rest of the pipeline (llm_extractor.py)
    doesn't need to know or care that these were batched together.
    """
    recognition_predictor, detection_predictor = _get_predictors()

    all_images: List[Image.Image] = []
    page_metadata: List[Tuple[str, int]] = []  # (pdf_path, page_number starting at 1)
    results: Dict[str, Dict[str, Any]] = {
        p: {"tables": [], "texts": [], "images": []} for p in pdf_paths
    }

    # Render every page of every PDF in this group first, all into one
    # flat list, remembering which (pdf, page number) each image came from.
    for pdf_path in pdf_paths:
        try:
            images = pdf_to_images(pdf_path, dpi=dpi)
        except Exception as e:
            logger.error(f"Could not render pages for {pdf_path}: {e}")
            continue
        results[pdf_path]["images"] = images
        for page_num, img in enumerate(images, start=1):
            all_images.append(img)
            page_metadata.append((pdf_path, page_num))

    if not all_images:
        logger.warning("No pages rendered for this OCR batch — nothing to OCR.")
        return results

    logger.info(f"Running OCR on {len(all_images)} page(s) across {len(pdf_paths)} PDF(s) "
                f"in ONE call (device {config.DEVICE})")

    # ONE call to the model for the whole group — this is the part that
    # actually fixes the "many threads fighting over one GPU" problem.
    page_preds = recognition_predictor(all_images, det_predictor=detection_predictor)

    # Sort the flat OCR results back out into their original per-PDF,
    # per-page buckets.
    for (pdf_path, page_num), page_pred in zip(page_metadata, page_preds):
        for para in _lines_to_paragraphs(page_pred):
            is_heading = para.isupper() and len(para) < 100
            results[pdf_path]["texts"].append({
                "type": "heading" if is_heading else "text",
                "text": para,
                "page": page_num,
            })

    for pdf_path, result in results.items():
        logger.info(f"OCR finished for {pdf_path}: {len(result['texts'])} text block(s) "
                    f"across {len(result['images'])} page(s)")

    return results


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