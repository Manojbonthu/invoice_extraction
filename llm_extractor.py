"""
llm_extractor.py - Sends invoice data to Gemini (main AI) to get structured
JSON out of it, checks the answer with rule_engine.py, and if any field
is still missing, sends the invoice PAGE IMAGE to Gemma (a vision AI that
can "look" at pictures) as a fallback.

WHAT'S NEW (in simple words):
  1. Uses proper logging now (logger) instead of print().
  2. Every AI call now goes through the RATE LIMITER from config.py first,
     so we never send more requests per minute than Google allows.
  3. Retries are smarter: only retries on "try again later" type errors
     (429 = too many requests, 503 = server busy, timeouts). Other errors
     (like a broken request) fail immediately instead of wasting 8 retries
     on something that will never succeed. Wait time between retries is
     now capped (never waits forever) and has small random "jitter" added
     (explained below).
  4. Uses STRUCTURED OUTPUT: instead of just asking the AI nicely to
     "please return JSON" and hoping, we tell Gemini the exact shape
     (schema) the answer must be in. This makes broken/malformed answers
     much rarer.
  5. Calls the new RuleEngine "document-level HSN" rule BEFORE the
     expensive Gemma image fallback — if that free rule already fixes the
     missing HSN, we skip paying for an extra AI image call.
  6. Gemma vision fallback now checks MULTIPLE PAGES if needed (not just
     page 1), using the "page" number we now store on each text block.
  7. Tracks how many tokens (pieces of text) and how much money each file
     costs, using config.calculate_cost(), so you can see real spend.

WHAT "jitter" MEANS (simple explanation):
  If 20 files all get rate-limited at the exact same moment, and they all
  wait exactly "4 seconds" before retrying, they'll all hit the server
  again at the exact same instant — causing another pile-up. Jitter just
  means adding a small random extra amount (like +0 to +1 second) so
  retries spread out naturally instead of retrying in one big wave.
"""

import re
import json
import time
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from rule_engine import RuleEngine
from digital_extractor import render_page_images

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# SCHEMA — the exact shape we want the AI's answer to be in.
# Passing this to Gemini as "response_schema" means Gemini is forced to
# answer in this shape, instead of us just hoping it follows instructions.
# ─────────────────────────────────────────────────────────

INVOICE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Invoice Number": {"type": "string", "nullable": True},
                    "Invoice Date": {"type": "string", "nullable": True},
                    "Total Payable": {"type": "number", "nullable": True},
                    "Total Tax": {"type": "number", "nullable": True},
                    "CGST": {"type": "number", "nullable": True},
                    "SGST": {"type": "number", "nullable": True},
                    "IGST": {"type": "number", "nullable": True},
                    "Invoice Items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_name": {"type": "string", "nullable": True},
                                "item_code": {"type": "string", "nullable": True},
                                "hsn_sac": {"type": "string", "nullable": True},
                                "quantity": {"type": "number", "nullable": True},
                                "rate": {"type": "number", "nullable": True},
                                "amount": {"type": "number", "nullable": True},
                            },
                        },
                    },
                },
            },
        }
    },
    "required": ["invoices"],
}

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT — the instructions we give the AI every time.
# ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert invoice data extraction assistant.

You will be given text extracted from an invoice (tables and text blocks).
Extract ALL invoices found into the required JSON schema.

RULES:

1. Invoice Number / Invoice Date / Total Payable: extract exactly as
   printed. If not found, use null — do not guess.

2. item_code: must be exactly 7 digits, and the first digit must be
   1, 2, 3, or 4 (i.e. between 1,000,000 and 4,999,999).
   - Prefer a field explicitly labeled "F.G Code" over "Input Code" if
     both appear.
   - Only if no labeled code exists, you may look in the item description
     for a standalone 7-digit number starting with 1-4 — but ONLY if it
     is not glued directly next to letters (e.g. do NOT extract "2500828"
     out of "502==F-2500828", since that is a reference/batch string, not
     an item code).

3. hsn_sac: must be exactly 4, 6, or 8 digits.
   - Do not confuse this with item_code (7 digits), Excise No, EWB No, or
     GST No — those are different fields with different digit counts.
   - Some invoices print ONE HSN code near the header/top of the page
     that applies to every item on the invoice, rather than one per row.
     If you see a single clearly-labeled "HSN Code: XXXXXXXX" near the
     top and no per-item HSN values, apply that same value to every item.

4. Total Tax: if CGST and SGST are both shown, Total Tax should be their
   sum. If only IGST is shown, Total Tax equals IGST.

5. If a field genuinely cannot be found anywhere in the given text,
   return null for that field. Do not invent values.

Return your answer using the exact JSON schema you were given — no extra
commentary, no markdown formatting, just the structured data.
"""

GEMMA_VISION_PROMPT_TEMPLATE = """You are looking at a photo/scan of an invoice page.
Extract the same invoice fields as described in these rules:

{rules}

This is a fallback pass — some fields were already found correctly by a
previous text-based pass. Do your best full, independent read of this
image using the same rules above.
"""


# ─────────────────────────────────────────────────────────
# USAGE TRACKING — remembers how many tokens / how much cost was used.
# run.py can read this after processing each file to save it into the
# tracker database.
# ─────────────────────────────────────────────────────────

_usage_totals = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}


def get_usage_totals() -> Dict[str, Any]:
    """Returns a copy of the running totals so far (tokens, cost, call count)."""
    return dict(_usage_totals)


def reset_usage_totals() -> None:
    """Call this before processing a new file, so totals reflect just that file."""
    _usage_totals.update({"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0})


def _record_usage(usage_metadata, model: str) -> None:
    """Reads token counts off the API response and adds them to our running totals."""
    if usage_metadata is None:
        return
    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    cost = config.calculate_cost(input_tokens, output_tokens, model=model)

    _usage_totals["input_tokens"] += input_tokens
    _usage_totals["output_tokens"] += output_tokens
    _usage_totals["cost"] += cost
    _usage_totals["calls"] += 1

    logger.debug(f"Call usage -> model={model}, input_tokens={input_tokens}, "
                 f"output_tokens={output_tokens}, cost=${cost:.6f}")


# ─────────────────────────────────────────────────────────
# BUILD THE TEXT PROMPT FROM EXTRACTED DATA
# ─────────────────────────────────────────────────────────

def format_data_for_llm(data: Dict[str, Any]) -> str:
    """
    Turns the {"tables": [...], "texts": [...]} dict (from either
    digital_extractor.py or scanned_extractor.py) into one labeled block
    of plain text that we send to the AI.
    """
    parts = []

    header_texts = [t["text"] for t in data.get("texts", []) if t.get("type") == "heading"]
    body_texts = [t["text"] for t in data.get("texts", []) if t.get("type") != "heading"]

    if header_texts:
        parts.append("=== INVOICE HEADER ===")
        parts.extend(header_texts)

    for i, table in enumerate(data.get("tables", []), start=1):
        parts.append(f"\n=== Table {i}: ITEM TABLE ===")
        headers = table.get("headers", [])
        if headers:
            parts.append(" | ".join(headers))
        for row in table.get("rows", []):
            parts.append(" | ".join(str(c) for c in row))

    if body_texts:
        parts.append("\n=== OTHER TEXT ===")
        parts.extend(body_texts)

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────
# CALL THE AI (with rate limiting + safe retries)
# ─────────────────────────────────────────────────────────

_RETRYABLE_STATUS_CODES = {429, 503}


def _is_retryable_error(exc: Exception) -> bool:
    """
    Decides if an error is worth retrying. 429 (too many requests) and 503
    (server temporarily busy) are worth retrying — the problem usually
    fixes itself after a short wait. Most other errors (like a bad
    request) will fail the same way every time, so retrying just wastes
    time.
    """
    message = str(exc).lower()
    if any(str(code) in message for code in _RETRYABLE_STATUS_CODES):
        return True
    if "timeout" in message or "timed out" in message or "deadline" in message:
        return True
    return False


def call_api(
    prompt: str,
    model: str = None,
    images: Optional[List] = None,
    max_retries: int = 5,
) -> Tuple[Optional[dict], Any]:
    """
    Sends a prompt (and optional images, for Gemma vision calls) to the AI
    and returns (parsed_json_dict, usage_metadata).

    Retries only on "temporary" errors (see _is_retryable_error), waiting
    longer each time (capped so it never waits forever), with a small
    random jitter added so many parallel workers don't all retry at the
    exact same second.
    """
    model = model or config.MODEL_NAME
    client = config.get_llm_client()

    is_gemma_call = "gemma" in model.lower()
    limiter = config.gemma_limiter if is_gemma_call else config.gemini_limiter

    contents = [prompt] if not images else [prompt, *images]

    last_exception = None
    for attempt in range(1, max_retries + 1):
        # Wait our turn under the rate limit before actually calling the API
        limiter.acquire()

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                    "response_schema": INVOICE_RESPONSE_SCHEMA,
                },
            )
            _record_usage(getattr(response, "usage_metadata", None), model)
            parsed = parse_response(response.text)
            return parsed, getattr(response, "usage_metadata", None)

        except Exception as e:
            last_exception = e
            if not _is_retryable_error(e):
                logger.error(f"Non-retryable error calling {model}: {e}")
                break

            if attempt >= max_retries:
                logger.error(f"Giving up on {model} after {attempt} attempts: {e}")
                break

            wait_time = min(2 ** attempt, 60) + random.uniform(0, 1)
            logger.warning(f"Retryable error from {model} (attempt {attempt}/{max_retries}): "
                            f"{e} — waiting {wait_time:.1f}s before retry")
            time.sleep(wait_time)

            # After 3 failed attempts on the main model, switch to the
            # smaller/cheaper fallback model for the remaining attempts.
            if attempt == 3 and model == config.MODEL_NAME:
                logger.info(f"Switching from {config.MODEL_NAME} to fallback model "
                            f"{config.FALLBACK_MODEL} after repeated failures")
                model = config.FALLBACK_MODEL

    logger.error(f"call_api failed permanently: {last_exception}")
    return None, None


def parse_response(response_text: str) -> Optional[dict]:
    """
    Turns the AI's text answer into a real Python dict. Even though we use
    structured output (which should already guarantee valid JSON), we keep
    this safety-net regex extraction in case some junk text (like markdown
    fences) sneaks in around the JSON.
    """
    if not response_text:
        return None
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.error("Could not parse JSON even after cleanup attempt.")
                return None
        logger.error("No JSON object found in AI response.")
        return None


# ─────────────────────────────────────────────────────────
# GEMMA VISION FALLBACK (multi-page aware)
# ─────────────────────────────────────────────────────────

def _find_null_fields(invoices: List[dict]) -> bool:
    """Returns True if ANY field (invoice-level or item-level) is still null."""
    top_level_fields = ["Invoice Number", "Invoice Date", "Total Payable", "Total Tax"]
    item_fields = ["item_name", "item_code", "hsn_sac", "quantity", "rate", "amount"]

    for inv in invoices:
        for field in top_level_fields:
            if inv.get(field) is None:
                return True
        for item in inv.get("Invoice Items", []):
            for field in item_fields:
                if item.get(field) is None:
                    return True
    return False


def _merge_only_nulls(original: List[dict], fallback: List[dict]) -> List[dict]:
    """
    Combines the original (Gemini) result with the fallback (Gemma) result.
    Matches invoices/items by their POSITION in the list (1st with 1st,
    2nd with 2nd, etc). For each field, ONLY fills it in if the original
    was null — never overwrites a value Gemini already got right.
    """
    for i, orig_inv in enumerate(original):
        if i >= len(fallback):
            logger.warning("Gemma returned fewer invoices than Gemini — skipping extras.")
            break
        fb_inv = fallback[i]

        for key, value in fb_inv.items():
            if key == "Invoice Items":
                continue
            if orig_inv.get(key) is None and value is not None:
                orig_inv[key] = value

        orig_items = orig_inv.get("Invoice Items", [])
        fb_items = fb_inv.get("Invoice Items", [])
        if len(fb_items) < len(orig_items):
            logger.warning("⚠️ Gemma returned fewer items than Gemini for this invoice — "
                            "extras will keep their original (possibly null) values.")

        for j, orig_item in enumerate(orig_items):
            if j >= len(fb_items):
                break
            fb_item = fb_items[j]
            for key, value in fb_item.items():
                if orig_item.get(key) is None and value is not None:
                    # Re-validate anything we're about to fill in for hsn_sac/item_code
                    if key == "hsn_sac" and not RuleEngine.validate_hsn(value):
                        continue
                    if key == "item_code" and not RuleEngine.validate_item_code(value):
                        continue
                    orig_item[key] = value

    return original


def apply_gemma_vision_fallback(invoices: List[dict], data: Dict[str, Any], pdf_path: Optional[str]) -> List[dict]:
    """
    If any field is still null after the text-based pass + rule checks,
    try showing the AI an actual PICTURE of the invoice page(s) instead
    of just text — sometimes a field is visually there but was missed or
    garbled during text/OCR extraction.

    MULTI-PAGE: tries page images one at a time (in order) until either
    every field is filled in, or we run out of pages to try. This
    replaces the old behavior of only ever checking page 1.
    """
    if not _find_null_fields(invoices):
        return invoices  # nothing missing — skip the extra AI call entirely

    images = data.get("images")  # already-rendered pages, for scanned PDFs
    if images is None:
        if not pdf_path:
            logger.warning("No page images available and no pdf_path given — "
                            "skipping Gemma vision fallback.")
            return invoices
        images = render_page_images(pdf_path)  # digital PDFs: render on demand

    if not images:
        logger.warning("No page images could be produced — skipping Gemma vision fallback.")
        return invoices

    rules_text = SYSTEM_PROMPT
    prompt = GEMMA_VISION_PROMPT_TEMPLATE.format(rules=rules_text)

    for page_index, page_image in enumerate(images, start=1):
        if not _find_null_fields(invoices):
            break  # already fixed everything, no need to check more pages

        logger.info(f"Gemma vision fallback: trying page {page_index} of {len(images)}")
        result, _usage = call_api(prompt, model=config.GEMMA_VISION_MODEL, images=[page_image])

        if not result or "invoices" not in result:
            logger.warning(f"Gemma returned no usable result for page {page_index}")
            continue

        invoices = _merge_only_nulls(invoices, result["invoices"])

    return invoices


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT — used by run.py and test_from_raw_json.py
# ─────────────────────────────────────────────────────────

def get_invoice_json_from_data(data: Dict[str, Any], pdf_path: Optional[str] = None) -> dict:
    """
    Full pipeline for ONE file's extracted data:
      1. Build the text prompt from tables/texts.
      2. Call Gemini to get structured JSON.
      3. Run RuleEngine checks (including the new document-level HSN rule).
      4. If anything is still missing, fall back to Gemma vision (image-based),
         trying multiple pages if needed.
      5. Return the final, validated result.

    Call get_usage_totals() after this function to see tokens/cost used
    for this file (call reset_usage_totals() first if you want totals
    per-file instead of cumulative across the whole run).
    """
    prompt = f"{SYSTEM_PROMPT}\n\n=== INVOICE DATA ===\n{format_data_for_llm(data)}"

    result, _usage = call_api(prompt, model=config.MODEL_NAME)
    if not result or "invoices" not in result:
        logger.error("Primary Gemini extraction failed or returned no invoices.")
        return {"invoices": []}

    invoices = result["invoices"]

    # Rule checks — includes the free, safe document-level HSN rule, which
    # runs BEFORE the costly Gemma vision fallback below.
    invoices = RuleEngine.apply_post_llm_rules(invoices, texts=data.get("texts"))

    # Only spend on Gemma vision if something is still missing after all that.
    invoices = apply_gemma_vision_fallback(invoices, data, pdf_path)

    # One more pass of the rule checks, in case Gemma filled in a field
    # that also needed the same validation (e.g. it found a new hsn_sac
    # that still needs format-checking).
    invoices = RuleEngine.apply_post_llm_rules(invoices, texts=data.get("texts"))

    usage = get_usage_totals()
    logger.info(f"Finished extraction: {len(invoices)} invoice(s), "
                f"{usage['calls']} AI call(s), cost so far ${usage['cost']:.6f}")

    return {"invoices": invoices}