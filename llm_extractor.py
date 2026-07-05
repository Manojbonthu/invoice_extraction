"""
llm_extractor.py - Sends invoice data to the active LLM provider (Gemini
OR OpenAI, picked by config.LLM_PROVIDER) to get structured JSON, checks
the answer with rule_engine.py, and if any field is still missing, sends
the invoice PAGE IMAGE to a vision-capable model as a fallback.

PROVIDER ABSTRACTION (NEW):
  This file no longer talks to the google-genai SDK directly. Instead:
    - call_api() is a thin dispatcher: it looks at config.LLM_PROVIDER
      and hands off to _call_gemini_backend() or _call_openai_backend().
    - Both backends return the exact same shape: (parsed_json, usage_dict).
    - Everything else in this file (format_data_for_llm, RuleEngine
      integration, the Gemma/GPT vision fallback loop, usage tracking)
      is 100% provider-agnostic — it was written once and works with
      whichever backend is active.
    - Model names come from config.PRIMARY_MODEL / config.FALLBACK_MODEL /
      config.VISION_MODEL, which config.py already resolved to the right
      provider's model names.

  This means: developing locally against Gemini and deploying against
  OpenAI in production requires ZERO changes to this file — only your
  .env's LLM_PROVIDER setting changes.

WHAT'S STILL THE SAME AS BEFORE:
  - Rate limiting before every call (config.get_limiter() picks the
    right limiter for whichever provider/model is active).
  - Smart retries: only on "try again later" errors (429/500/502/503,
    timeouts), capped wait time, small random jitter.
  - Structured output: Gemini uses response_schema, OpenAI uses strict
    JSON Schema mode (response_format={"type": "json_schema", ...}) —
    both force the model into the exact shape we need.
  - Document-level HSN rule runs before the (paid) vision fallback.
  - Multi-page vision fallback (tries page images in order until fields
    are filled or pages run out).
  - Per-file usage/cost tracking returned as a LOCAL value from
    get_invoice_json_from_data() — NOT a shared global — so this is
    still safe to call from many worker threads at once.
"""

import re
import io
import json
import time
import base64
import random
import logging
from typing import Any, Dict, List, Optional, Tuple

import config
from rule_engine import RuleEngine
from digital_extractor import render_page_images

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# SCHEMAS — one per provider, since Gemini and OpenAI express
# "nullable"/"optional" fields slightly differently in their structured
# output formats.
# ─────────────────────────────────────────────────────────

# Gemini's response_schema style: "nullable": True alongside "type".
GEMINI_RESPONSE_SCHEMA = {
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

# OpenAI's "strict" structured output mode requires: every property
# listed in "required" (even nullable ones, expressed as type: [X, "null"]),
# and "additionalProperties": false on every object level.
OPENAI_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "invoices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "Invoice Number": {"type": ["string", "null"]},
                    "Invoice Date": {"type": ["string", "null"]},
                    "Total Payable": {"type": ["number", "null"]},
                    "Total Tax": {"type": ["number", "null"]},
                    "CGST": {"type": ["number", "null"]},
                    "SGST": {"type": ["number", "null"]},
                    "IGST": {"type": ["number", "null"]},
                    "Invoice Items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "item_name": {"type": ["string", "null"]},
                                "item_code": {"type": ["string", "null"]},
                                "hsn_sac": {"type": ["string", "null"]},
                                "quantity": {"type": ["number", "null"]},
                                "rate": {"type": ["number", "null"]},
                                "amount": {"type": ["number", "null"]},
                            },
                            "required": ["item_name", "item_code", "hsn_sac",
                                         "quantity", "rate", "amount"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["Invoice Number", "Invoice Date", "Total Payable", "Total Tax",
                             "CGST", "SGST", "IGST", "Invoice Items"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["invoices"],
    "additionalProperties": False,
}

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT — the instructions we give the AI every time. Same
# prompt for both providers; it's provider-agnostic text.
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

VISION_PROMPT_TEMPLATE = """You are looking at a photo/scan of an invoice page.
Extract the same invoice fields as described in these rules:

{rules}

This is a fallback pass — some fields were already found correctly by a
previous text-based pass. Do your best full, independent read of this
image using the same rules above.
"""


# ─────────────────────────────────────────────────────────
# USAGE TRACKING — plain local accumulator, NOT a shared global. Every
# function below that talks to the AI returns its own usage dict;
# callers add it into their own local total. Nothing here is shared
# across threads, so concurrent files can never corrupt each other's
# token/cost numbers.
# ─────────────────────────────────────────────────────────

def _new_usage() -> Dict[str, Any]:
    """A fresh, empty usage accumulator. Call this once per file."""
    return {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "calls": 0}


def _add_usage(total: Dict[str, Any], addition: Optional[Dict[str, Any]]) -> None:
    """Adds one call's usage into a running total (both plain local dicts)."""
    if not addition:
        return
    total["input_tokens"] += addition.get("input_tokens", 0)
    total["output_tokens"] += addition.get("output_tokens", 0)
    total["cost"] += addition.get("cost", 0.0)
    total["calls"] += addition.get("calls", 0)


def _usage_from_gemini(usage_metadata, model: str) -> Dict[str, Any]:
    """Turns one Gemini response's usage_metadata into a plain usage dict."""
    if usage_metadata is None:
        return _new_usage()
    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    cost = config.calculate_cost(input_tokens, output_tokens, model=model)
    logger.debug(f"Call usage -> model={model}, input_tokens={input_tokens}, "
                 f"output_tokens={output_tokens}, cost=${cost:.6f}")
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost, "calls": 1}


def _usage_from_openai(usage_obj, model: str) -> Dict[str, Any]:
    """Turns one OpenAI response's usage object into a plain usage dict."""
    if usage_obj is None:
        return _new_usage()
    input_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    cost = config.calculate_cost(input_tokens, output_tokens, model=model)
    logger.debug(f"Call usage -> model={model}, input_tokens={input_tokens}, "
                 f"output_tokens={output_tokens}, cost=${cost:.6f}")
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "cost": cost, "calls": 1}


# ─────────────────────────────────────────────────────────
# BUILD THE TEXT PROMPT FROM EXTRACTED DATA (unchanged, provider-agnostic)
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
# IMAGE HELPERS — Gemini's SDK accepts PIL images directly; OpenAI's
# chat.completions API wants a base64 data URL instead.
# ─────────────────────────────────────────────────────────

def _pil_to_data_url(image) -> str:
    """Encodes a PIL image as a base64 PNG data URL for OpenAI's vision input."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ─────────────────────────────────────────────────────────
# RETRY DECISION — shared by both backends.
# ─────────────────────────────────────────────────────────

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}


def _is_retryable_error(exc: Exception) -> bool:
    """
    Decides if an error is worth retrying. 429/500/502/503 and timeouts
    are worth retrying — the problem usually fixes itself after a short
    wait. Most other errors (like a malformed request) will fail the
    same way every time, so retrying just wastes time.
    """
    message = str(exc).lower()
    if any(str(code) in message for code in _RETRYABLE_STATUS_CODES):
        return True
    if "timeout" in message or "timed out" in message or "deadline" in message:
        return True
    return False


def parse_response(response_text: str) -> Optional[dict]:
    """
    Turns the AI's text answer into a real Python dict. Even with
    structured output (which should already guarantee valid JSON), we
    keep this safety-net regex extraction in case junk text (like
    markdown fences) sneaks in around the JSON.
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
# PROVIDER BACKEND: GEMINI
# ─────────────────────────────────────────────────────────

def _call_gemini_backend(
    prompt: str,
    model: str,
    images: Optional[List],
    max_retries: int,
) -> Tuple[Optional[dict], Dict[str, Any]]:
    client = config.get_llm_client()
    limiter = config.get_limiter(model)
    contents = [prompt] if not images else [prompt, *images]

    last_exception = None
    for attempt in range(1, max_retries + 1):
        limiter.acquire()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={
                    "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                    "response_schema": GEMINI_RESPONSE_SCHEMA,
                },
            )
            usage = _usage_from_gemini(getattr(response, "usage_metadata", None), model)
            parsed = parse_response(response.text)
            return parsed, usage

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

            if attempt == 3 and model == config.PRIMARY_MODEL:
                logger.info(f"Switching from {config.PRIMARY_MODEL} to fallback model "
                            f"{config.FALLBACK_MODEL} after repeated failures")
                model = config.FALLBACK_MODEL

    logger.error(f"call_api (gemini) failed permanently: {last_exception}")
    return None, _new_usage()


# ─────────────────────────────────────────────────────────
# PROVIDER BACKEND: OPENAI
# ─────────────────────────────────────────────────────────

def _call_openai_backend(
    prompt: str,
    model: str,
    images: Optional[List],
    max_retries: int,
) -> Tuple[Optional[dict], Dict[str, Any]]:
    client = config.get_llm_client()
    limiter = config.get_limiter(model)

    if images:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    last_exception = None
    for attempt in range(1, max_retries + 1):
        limiter.acquire()
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=config.MAX_OUTPUT_TOKENS,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "invoice_extraction",
                        "strict": True,
                        "schema": OPENAI_RESPONSE_SCHEMA,
                    },
                },
            )
            usage = _usage_from_openai(getattr(response, "usage", None), model)
            parsed = parse_response(response.choices[0].message.content)
            return parsed, usage

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

            if attempt == 3 and model == config.PRIMARY_MODEL:
                logger.info(f"Switching from {config.PRIMARY_MODEL} to fallback model "
                            f"{config.FALLBACK_MODEL} after repeated failures")
                model = config.FALLBACK_MODEL

    logger.error(f"call_api (openai) failed permanently: {last_exception}")
    return None, _new_usage()


# ─────────────────────────────────────────────────────────
# DISPATCHER — the ONLY place that knows which provider is active.
# Everything else in this file (and every OTHER file in the project)
# just calls call_api() the same way regardless of provider.
# ─────────────────────────────────────────────────────────

def call_api(
    prompt: str,
    model: str = None,
    images: Optional[List] = None,
    max_retries: int = 5,
) -> Tuple[Optional[dict], Dict[str, Any]]:
    """
    Sends a prompt (and optional images) to whichever LLM provider is
    active (config.LLM_PROVIDER) and returns (parsed_json_dict, usage_dict).
    usage_dict is THIS call's usage only — a plain local dict, safe to
    use from any number of worker threads at once.
    """
    model = model or config.PRIMARY_MODEL
    if config.LLM_PROVIDER == "openai":
        return _call_openai_backend(prompt, model, images, max_retries)
    return _call_gemini_backend(prompt, model, images, max_retries)


# ─────────────────────────────────────────────────────────
# VISION FALLBACK (multi-page aware, provider-agnostic)
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
    Combines the original result with the vision-fallback result.
    Matches invoices/items by POSITION in the list. For each field, ONLY
    fills it in if the original was null — never overwrites a value the
    primary pass already got right.
    """
    for i, orig_inv in enumerate(original):
        if i >= len(fallback):
            logger.warning("Vision fallback returned fewer invoices than the primary pass — "
                            "skipping extras.")
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
            logger.warning("⚠️ Vision fallback returned fewer items than the primary pass for "
                            "this invoice — extras will keep their original (possibly null) values.")

        for j, orig_item in enumerate(orig_items):
            if j >= len(fb_items):
                break
            fb_item = fb_items[j]
            for key, value in fb_item.items():
                if orig_item.get(key) is None and value is not None:
                    if key == "hsn_sac" and not RuleEngine.validate_hsn(value):
                        continue
                    if key == "item_code" and not RuleEngine.validate_item_code(value):
                        continue
                    orig_item[key] = value

    return original


def apply_vision_fallback(
    invoices: List[dict],
    data: Dict[str, Any],
    pdf_path: Optional[str],
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    If any field is still null after the text-based pass + rule checks,
    show the AI an actual PICTURE of the invoice page(s) — sometimes a
    field is visually there but was missed or garbled during text/OCR
    extraction. Uses config.VISION_MODEL, which is Gemma or GPT-4o(-mini)
    depending on config.LLM_PROVIDER.

    MULTI-PAGE: tries page images one at a time (in order) until either
    every field is filled in, or pages run out.

    Returns (invoices, usage) — usage covers only the vision calls made
    in this function call, for this one file.
    """
    usage = _new_usage()

    if not _find_null_fields(invoices):
        return invoices, usage  # nothing missing — skip the extra AI call entirely

    images = data.get("images")  # already-rendered pages, for scanned PDFs
    if images is None:
        if not pdf_path:
            logger.warning("No page images available and no pdf_path given — "
                            "skipping vision fallback.")
            return invoices, usage
        images = render_page_images(pdf_path)  # digital PDFs: render on demand

    if not images:
        logger.warning("No page images could be produced — skipping vision fallback.")
        return invoices, usage

    prompt = VISION_PROMPT_TEMPLATE.format(rules=SYSTEM_PROMPT)

    for page_index, page_image in enumerate(images, start=1):
        if not _find_null_fields(invoices):
            break  # already fixed everything, no need to check more pages

        logger.info(f"Vision fallback ({config.LLM_PROVIDER}/{config.VISION_MODEL}): "
                    f"trying page {page_index} of {len(images)}")
        result, call_usage = call_api(prompt, model=config.VISION_MODEL, images=[page_image])
        _add_usage(usage, call_usage)

        if not result or "invoices" not in result:
            logger.warning(f"Vision fallback returned no usable result for page {page_index}")
            continue

        invoices = _merge_only_nulls(invoices, result["invoices"])

    return invoices, usage


# ─────────────────────────────────────────────────────────
# MAIN ENTRY POINT — used by run.py and test scripts
# ─────────────────────────────────────────────────────────

def get_invoice_json_from_data(
    data: Dict[str, Any],
    pdf_path: Optional[str] = None,
) -> Tuple[dict, Dict[str, Any]]:
    """
    Full pipeline for ONE file's extracted data:
      1. Build the text prompt from tables/texts.
      2. Call the active provider's primary model to get structured JSON.
      3. Run RuleEngine checks (including the document-level HSN rule).
      4. If anything is still missing, fall back to vision (image-based),
         trying multiple pages if needed.
      5. Return the final, validated result.

    Returns (result, usage) — usage is a LOCAL accumulator for just this
    one call (this one file), covering every AI call made along the way.
    Safe to call from multiple threads at once: nothing here is shared
    global state, and it works identically whether config.LLM_PROVIDER
    is "gemini" or "openai".
    """
    usage = _new_usage()

    prompt = f"{SYSTEM_PROMPT}\n\n=== INVOICE DATA ===\n{format_data_for_llm(data)}"

    result, call_usage = call_api(prompt, model=config.PRIMARY_MODEL)
    _add_usage(usage, call_usage)

    if not result or "invoices" not in result:
        logger.error("Primary extraction failed or returned no invoices.")
        return {"invoices": []}, usage

    invoices = result["invoices"]

    invoices = RuleEngine.apply_post_llm_rules(invoices, texts=data.get("texts"))

    invoices, fallback_usage = apply_vision_fallback(invoices, data, pdf_path)
    _add_usage(usage, fallback_usage)

    invoices = RuleEngine.apply_post_llm_rules(invoices, texts=data.get("texts"))

    logger.info(f"Finished extraction ({config.LLM_PROVIDER}): {len(invoices)} invoice(s), "
                f"{usage['calls']} AI call(s), cost for this file ${usage['cost']:.6f}")

    return {"invoices": invoices}, usage