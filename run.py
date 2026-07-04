"""
run.py - Runs the FULL pipeline across many PDF files (batch processing).

WHAT'S NEW (in simple words):
  1. Uses proper logging now (logger) instead of print().
  2. RESUME SUPPORT: keeps a small database file (run_status.db) that
     remembers which files are already done. If the program crashes or
     you stop it halfway through 100,000 files, running it again will
     SKIP the files already finished, instead of starting over from zero.
  3. WORKER POOL: processes multiple files AT THE SAME TIME (instead of
     one by one), using config.MAX_WORKERS. This makes big batches much
     faster, since most of the waiting time is just waiting for the AI's
     reply — while one file waits, another file can be working.
  4. STARTUP CHECK: tests your AI model names before starting the batch,
     so a typo in a model name fails immediately instead of after
     thousands of files.
  5. NEW COMMAND OPTIONS:
       --workers N     how many files to process at the same time
       --resume        skip files already completed (this is the default)
       --force         reprocess EVERY file, even ones already completed
       --limit N       only process the first N files (good for testing)
  6. Tracks tokens used and cost spent per file, saved into the database.
"""

import sys
import time
import json
import sqlite3
import logging
import argparse
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from detector import detect_pdf_type
from digital_extractor import extract_digital_clean
from scanned_extractor import extract_scanned_clean
from llm_extractor import get_invoice_json_from_data, get_usage_totals, reset_usage_totals

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
OUTPUT_FOLDER = Path(config.OUTPUT_DIR)
JSON_FOLDER = OUTPUT_FOLDER / "JSON"
JSON_RAW_FOLDER = OUTPUT_FOLDER / "JSON_Raw"

# One shared lock protects writes to the SQLite database, since many
# worker threads may finish files at almost the same moment.
_db_lock = threading.Lock()


# ─────────────────────────────────────────────────────────
# STATUS TRACKING (built into this file, using SQLite — no extra file)
# This is what makes "resume" possible: every file's outcome is written
# here, so a second run can check "have I already done this one?"
# ─────────────────────────────────────────────────────────

def _get_db_connection():
    """Opens a connection to the tracking database, creating the table
    if it doesn't exist yet."""
    conn = sqlite3.connect(config.DB_PATH, timeout=30)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS file_status (
            doc_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
            attempts INTEGER DEFAULT 0,
            error TEXT,
            tokens_used INTEGER DEFAULT 0,
            cost REAL DEFAULT 0.0,
            started_at TEXT,
            finished_at TEXT
        )
    """)
    conn.commit()
    return conn


def _is_already_done(doc_id: str) -> bool:
    """Checks if this file was already successfully processed before."""
    with _db_lock:
        conn = _get_db_connection()
        try:
            row = conn.execute(
                "SELECT status FROM file_status WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            return row is not None and row[0] == "success"
        finally:
            conn.close()


def _mark_started(doc_id: str, filename: str) -> None:
    with _db_lock:
        conn = _get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                INSERT INTO file_status (doc_id, filename, status, attempts, started_at)
                VALUES (?, ?, 'in_progress', 1, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status = 'in_progress',
                    attempts = attempts + 1,
                    started_at = excluded.started_at
            """, (doc_id, filename, now))
            conn.commit()
        finally:
            conn.close()


def _mark_finished(doc_id: str, status: str, error: str = None,
                    tokens_used: int = 0, cost: float = 0.0) -> None:
    with _db_lock:
        conn = _get_db_connection()
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                UPDATE file_status
                SET status = ?, error = ?, tokens_used = ?, cost = ?, finished_at = ?
                WHERE doc_id = ?
            """, (status, error, tokens_used, cost, now, doc_id))
            conn.commit()
        finally:
            conn.close()


def print_summary() -> None:
    """Shows an overview of the whole batch: how many succeeded, failed,
    total tokens used, total cost so far. Simple version of what a
    separate tracker.py would have done."""
    conn = _get_db_connection()
    try:
        rows = conn.execute("""
            SELECT status, COUNT(*), COALESCE(SUM(tokens_used), 0), COALESCE(SUM(cost), 0)
            FROM file_status GROUP BY status
        """).fetchall()

        logger.info("=" * 60)
        logger.info("BATCH STATUS SUMMARY")
        total_cost = 0.0
        for status, count, tokens, cost in rows:
            logger.info(f"  {status}: {count} file(s), {tokens} tokens, ${cost:.4f}")
            total_cost += cost
        logger.info(f"  TOTAL COST SO FAR: ${total_cost:.4f}")
        logger.info("=" * 60)

        failures = conn.execute(
            "SELECT filename, error FROM file_status WHERE status = 'failed'"
        ).fetchall()
        if failures:
            logger.info(f"Failed files ({len(failures)}):")
            for filename, error in failures:
                logger.info(f"  - {filename}: {error}")
    finally:
        conn.close()


# ─────────────────────────────────────────────────────────
# COLLECT PDFs TO PROCESS
# ─────────────────────────────────────────────────────────

def collect_pdfs(root_dir: str):
    root = Path(root_dir)
    pdfs = []
    for path in root.rglob("*.pdf"):
        if "processed" in path.parts or "Job Work" in path.parts:
            continue
        if "1.Input" in path.parts:
            pdfs.append(path)
    return sorted(pdfs)


# ─────────────────────────────────────────────────────────
# PROCESS ONE FILE (same logic as before, now with tracking + logging)
# ─────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path) -> bool:
    fname = pdf_path.name
    doc_id = pdf_path.stem
    logger.info(f"Processing: {fname}")

    _mark_started(doc_id, fname)
    reset_usage_totals()  # so tokens/cost recorded below are just for THIS file

    # 1. Detect type
    try:
        overall, _ = detect_pdf_type(str(pdf_path))
        logger.info(f"{fname}: detected as {overall}")
    except Exception as e:
        logger.error(f"{fname}: detection failed: {e}")
        _mark_finished(doc_id, "failed", error=f"detection failed: {e}")
        return False

    # 2. Extract (dict schema, same for both paths)
    try:
        if overall == "digital":
            data = extract_digital_clean(str(pdf_path))
        else:
            data = extract_scanned_clean(str(pdf_path))
    except Exception as e:
        logger.error(f"{fname}: extraction failed: {e}")
        _mark_finished(doc_id, "failed", error=f"extraction failed: {e}")
        return False

    if not data.get("tables") and not data.get("texts"):
        logger.warning(f"{fname}: no data extracted")
        _mark_finished(doc_id, "failed", error="no data extracted")
        return False

    logger.info(f"{fname}: extracted {len(data.get('tables', []))} tables, "
                f"{len(data.get('texts', []))} text blocks")

    # 3. Save raw extraction (exclude "images" — PIL images aren't JSON-serializable)
    try:
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        JSON_RAW_FOLDER.mkdir(parents=True, exist_ok=True)
        raw_path = JSON_RAW_FOLDER / f"{doc_id}.json"
        raw_for_disk = {k: v for k, v in data.items() if k != "images"}
        raw_path.write_text(json.dumps(raw_for_disk, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"{fname}: saved raw JSON -> {raw_path}")
    except Exception as e:
        logger.warning(f"{fname}: could not save raw JSON (continuing anyway): {e}")

    # 4. Send dict to Gemini (+ Gemma vision fallback if needed)
    try:
        result = get_invoice_json_from_data(data, pdf_path=str(pdf_path))
    except Exception as e:
        logger.error(f"{fname}: LLM extraction failed: {e}")
        usage = get_usage_totals()
        _mark_finished(doc_id, "failed", error=f"LLM extraction failed: {e}",
                        tokens_used=usage["input_tokens"] + usage["output_tokens"],
                        cost=usage["cost"])
        return False

    # 5. Save clean JSON
    try:
        JSON_FOLDER.mkdir(parents=True, exist_ok=True)
        json_path = JSON_FOLDER / f"{doc_id}.json"
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"{fname}: saved final JSON -> {json_path}")
    except Exception as e:
        logger.error(f"{fname}: failed to save final JSON: {e}")
        usage = get_usage_totals()
        _mark_finished(doc_id, "failed", error=f"failed to save output: {e}",
                        tokens_used=usage["input_tokens"] + usage["output_tokens"],
                        cost=usage["cost"])
        return False

    usage = get_usage_totals()
    _mark_finished(doc_id, "success",
                    tokens_used=usage["input_tokens"] + usage["output_tokens"],
                    cost=usage["cost"])
    return True


# ─────────────────────────────────────────────────────────
# MAIN — parses command-line options and runs the batch
# ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Invoice extraction batch pipeline")
    parser.add_argument("single_file", nargs="?", default=None,
                         help="Process just one PDF file instead of the whole batch")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS,
                         help="How many files to process at the same time")
    parser.add_argument("--force", action="store_true",
                         help="Reprocess every file, even ones already marked successful")
    parser.add_argument("--resume", action="store_true", default=True,
                         help="Skip files already completed (this is the default behavior)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N files (useful for a quick test)")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("INVOICE EXTRACTION PIPELINE — BATCH MODE")
    logger.info(f"Device: {config.DEVICE} | Workers: {args.workers} | "
                f"OCR workers: {config.OCR_WORKERS} | OCR batch size: {config.OCR_BATCH_SIZE}")
    logger.info("=" * 60)

    # Check that model names actually work BEFORE processing thousands of files
    logger.info("Validating AI models before starting...")
    config.validate_models()
    logger.info("Model validation passed.")

    if args.single_file:
        pdfs = [Path(args.single_file)]
        if not pdfs[0].exists():
            logger.error(f"File not found: {pdfs[0]}")
            return
    else:
        pdfs = collect_pdfs(config.INPUT_DIR)
        if not pdfs:
            logger.warning(f"No PDFs found in {config.INPUT_DIR} (looking for 1.Input folders).")
            return

    # Skip already-completed files unless --force was passed
    if not args.force:
        before_count = len(pdfs)
        pdfs = [p for p in pdfs if not _is_already_done(p.stem)]
        skipped = before_count - len(pdfs)
        if skipped:
            logger.info(f"Skipping {skipped} file(s) already marked successful "
                        f"(use --force to reprocess them anyway)")

    if args.limit:
        pdfs = pdfs[:args.limit]
        logger.info(f"Limiting this run to the first {len(pdfs)} file(s)")

    if not pdfs:
        logger.info("Nothing to process — all files already completed. Use --force to redo them.")
        print_summary()
        return

    logger.info(f"Starting batch: {len(pdfs)} file(s) to process, {args.workers} worker(s)")

    total_start = time.time()
    success = 0
    failed = 0

    # Process multiple files at the same time using a worker pool.
    # This helps a lot because most of the time is spent WAITING for the
    # AI's response — while one file is waiting, another file's turn can
    # start, instead of everything waiting in a single line.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf): pdf for pdf in pdfs}

        for future in as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                ok = future.result()
            except Exception as e:
                logger.error(f"{pdf.name}: unexpected worker error: {e}")
                ok = False

            if ok:
                success += 1
            else:
                failed += 1

            done_count = success + failed
            logger.info(f"Progress: {done_count}/{len(pdfs)} "
                        f"(success: {success}, failed: {failed})")

    total_time = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"BATCH COMPLETE — Success: {success}, Failed: {failed}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info("=" * 60)

    print_summary()


if __name__ == "__main__":
    main()