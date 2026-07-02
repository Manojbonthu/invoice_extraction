"""
llm_extractor.py – Gemini with fallback model, retry logic, and robust JSON parsing.
UPDATED: System prompt with explicit item_code labels, priority, and fallback rules.
"""

import os
import json
import re
import time
from typing import Dict, Any

from config import get_llm_client, MODEL_NAME, MAX_OUTPUT_TOKENS, REQUEST_DELAY

FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")

# ─── UPDATED SYSTEM PROMPT ──────────────────────────────────────────
SYSTEM_PROMPT = """You are an invoice extractor. Output JSON matching the schema below.

If multiple distinct Invoice Numbers exist, return an array of objects.
Otherwise, return a single object.

Schema:
{
  "Invoice Number": string or null,
  "Invoice Date": "dd/mm/yyyy" or null,
  "Total Payable Amount": float or null,
  "Total Tax": float or null,
  "Invoice Items": [
    {
      "item_name": string or null,
      "hsn_sac": int or null,   // exactly 4,6,8 digits
      "rate": float or null,
      "quantity": float or null,
      "item_code": int or null  // 7 digits, 1000000-4999999
    }
  ]
}

Extraction Rules:

-1. Invoice Number:
   - Look for labels: "Invoice No", "SAPINV", "Invoice Number", "Inv No".
   - Keep the full string (e.g., "SAPINV/000798/26").
   - If multiple numbers exist, pick the one explicitly labelled as invoice.
   - Never use "LR No.", "Reference No.", "Order No.", or "Document No." unless no invoice label exists.
- **Invoice Date**: Find in header, format dd/mm/yyyy.

- **Total Payable**: Find in footer via "Grand Total", "Net Amount", "Total Payable". Use largest if multiple.

- **Total Tax**: Find in footer via "Total Tax", "IGST Total", or sum CGST+SGST if given separately.
  If not explicitly provided, sum CGST and SGST (or use IGST) when available.

- **Items**: From the ITEM TABLE. One object per row.

- **item_code**: Hunt for a 7-digit number (1000000-4999999) using labels:
  Product Code, Cust Item, MATERIAL CODE, PART NO, FG CODE, ITEM CODE, SKU, Input Code, Material Code, Cust. Item, Part No.
  Priority: If both FG Code and Input Code exist, choose FG Code (finished product). If only Input Code, use that.
  If none of those labels are found, scan the item_name/goods description for any 7-digit number starting with 1-4 and extract that as item_code.
  Extract if found, else null.

-- **hsn_sac**: Exactly 4,6,8 digits. Do not confuse with item_code. Look for labels: "HSN/SAC", "SAC CODE", "SAC", "HSN", "HSN Code or scan the item row for a 4/6/8 digit number.

All keys must be present. Use null for missing values. Numeric fields must be numbers, not strings.
Output ONLY valid JSON. No markdown, no explanations.
"""

def format_data_for_llm(data: Dict[str, Any]) -> str:
    """Format data with ALL tables clearly labelled."""
    parts = []
    if data.get("texts"):
        parts.append("=== INVOICE HEADER ===")
        for txt in data["texts"]:
            parts.append(txt.get("text", "") if isinstance(txt, dict) else str(txt))
        parts.append("")
    if data.get("tables"):
        for idx, tbl in enumerate(data["tables"], 1):
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            header_text = " ".join(str(h).lower() for h in headers)
            if "qty" in header_text or "rate" in header_text or "amount" in header_text:
                label = f"Table {idx}: ITEM TABLE"
            elif "tax" in header_text or "gst" in header_text:
                label = f"Table {idx}: TAX / TOTALS"
            else:
                label = f"Table {idx}: GENERAL"
            parts.append(f"=== {label} ===")
            if headers:
                parts.append(" | ".join(str(h) for h in headers if h))
                parts.append("-" * 80)
            for row in rows:
                clean_row = [str(c).replace("\n", " ").strip() for c in row if c is not None]
                if clean_row:
                    parts.append(" | ".join(clean_row))
            parts.append("")
    return "\n".join(parts)

def call_api(prompt: str, retries: int = 8):
    client = get_llm_client()
    last_exception = None
    for attempt in range(retries):
        model_to_use = MODEL_NAME if attempt < 3 else FALLBACK_MODEL
        try:
            response = client.models.generate_content(
                model=model_to_use,
                contents=f"{SYSTEM_PROMPT}\n\n{prompt}",
                config={
                    "temperature": 0.0,
                    "max_output_tokens": MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                    "thinking_config": {"thinking_budget": 0},
                }
            )
            # Capture real token usage from Gemini's response metadata
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            thinking_tokens = getattr(usage, "thoughts_token_count", 0) if usage else 0
            total_tokens = getattr(usage, "total_token_count", 0) if usage else 0

            print(f"Tokens - input: {input_tokens}, output: {output_tokens}, "
                  f"thinking: {thinking_tokens}, total: {total_tokens}")

            class WrappedResponse:
                def __init__(self, text, input_tok, output_tok, total_tok):
                    self.choices = [type('Choice', (), {'message': type('Message', (), {'content': text})()})()]
                    self.input_tokens = input_tok
                    self.output_tokens = output_tok
                    self.total_tokens = total_tok
            return WrappedResponse(response.text, input_tokens, output_tokens, total_tokens)
        except Exception as e:
            last_exception = e
            if "503" in str(e) or "429" in str(e):
                wait_time = (2 ** attempt) * 2
                print(f"⏳ Model busy ({model_to_use}), retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                time.sleep(REQUEST_DELAY)
    raise last_exception or Exception("All retries failed")

def parse_response(response):
    content = response.choices[0].message.content.strip()
    content = re.sub(r'^```json\s*', '', content)
    content = re.sub(r'\s*```$', '', content)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except:
                raise ValueError(f"Invalid JSON from LLM: {content[:200]}")
        else:
            raise ValueError(f"Invalid JSON from LLM: {content[:200]}")
    return parsed, response.input_tokens, response.output_tokens, response.total_tokens

def extract_invoice_from_clean_data(data: Dict[str, Any], filename: str) -> bool:
    doc_id = os.path.splitext(os.path.basename(filename))[0]
    if not data.get("tables") and not data.get("texts"):
        print(f"⚠️ No data for {filename}")
        return False
    try:
        prompt = format_data_for_llm(data)
        response = call_api(prompt)
        extracted, in_tok, out_tok, tot_tok = parse_response(response)
        os.makedirs("data/parsed", exist_ok=True)
        out_path = os.path.join("data/parsed", f"{doc_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(extracted, f, indent=2, ensure_ascii=False)
        if isinstance(extracted, list):
            print(f"✅ {filename}: extracted {len(extracted)} invoices")
            print(f"   Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")
        else:
            print(f"✅ {filename}: extraction successful (1 invoice)")
            print(f"   Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")
        return True
    except Exception as e:
        print(f"❌ {filename} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_invoice_json_from_data(data: Dict[str, Any]):
    """
    Send dict data (tables + texts, from extract_digital_clean /
    extract_scanned_clean) to the LLM via format_data_for_llm, validate
    with RuleEngine, return the result. Does NOT save to disk.
    """
    from rule_engine import RuleEngine

    if not data.get("tables") and not data.get("texts"):
        raise ValueError("Empty data passed to LLM (no tables or texts)")

    prompt = format_data_for_llm(data)
    response = call_api(prompt)
    extracted, in_tok, out_tok, tot_tok = parse_response(response)
    print(f"Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")

    was_list = isinstance(extracted, list)
    invoices = extracted if was_list else [extracted]
    invoices = RuleEngine.apply_post_llm_rules(invoices)
    return invoices if was_list else invoices[0]


def get_invoice_json(text: str):
    from rule_engine import RuleEngine
    if not text or not text.strip():
        raise ValueError("Empty text passed to LLM")
    response = call_api(text)
    extracted, in_tok, out_tok, tot_tok = parse_response(response)
    print(f"Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")
    was_list = isinstance(extracted, list)
    invoices = extracted if was_list else [extracted]
    invoices = RuleEngine.apply_post_llm_rules(invoices)
    return invoices if was_list else invoices[0]

def extract_invoice_from_text(text: str, filename: str) -> bool:
    from rule_engine import RuleEngine
    doc_id = os.path.splitext(os.path.basename(filename))[0]
    if not text or not text.strip():
        print(f"⚠️ No text content for {filename}")
        return False
    try:
        response = call_api(text)
        extracted, in_tok, out_tok, tot_tok = parse_response(response)
        was_list = isinstance(extracted, list)
        invoices = extracted if was_list else [extracted]
        invoices = RuleEngine.apply_post_llm_rules(invoices)
        final = invoices if was_list else invoices[0]
        os.makedirs("data/parsed", exist_ok=True)
        out_path = os.path.join("data/parsed", f"{doc_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final, f, indent=2, ensure_ascii=False)
        if was_list:
            print(f"✅ {filename}: extracted {len(final)} invoices")
            print(f"   Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")
        else:
            print(f"✅ {filename}: extraction successful (1 invoice)")
            print(f"   Tokens used - input: {in_tok}, output: {out_tok}, total: {tot_tok}")
        return True
    except Exception as e:
        print(f"❌ {filename} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        extract_invoice_from_clean_data(data, json_path)
    else:
        print("Usage: python llm_extractor.py <path_to_clean_json>")