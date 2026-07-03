"""
rule_engine.py – Invoice validation rules.
UPDATED:
  - Added total tax computation from CGST+SGST.
  - REMOVED the regex-based HSN-from-item_name fallback. It was too
    permissive — it grabbed ANY standalone 4/6/8-digit number from the
    item description as if it were an HSN, with no check that it was
    actually labeled as one. On vendor formats where the item description
    contains embedded batch/quantity numbers (e.g. "3900==F-2005134"), this
    caused false positives (hsn_sac wrongly set to "3900") that then
    prevented the Gemma vision fallback from ever being triggered for that
    field, since it was no longer null.
  - extract_hsn_from_text() is kept defined below (harmless, unused) in
    case it's useful for a more targeted use later, but apply_post_llm_rules()
    no longer calls it automatically.
"""

import re
import logging

logger = logging.getLogger(__name__)

class RuleEngine:
    """Static validation methods for invoice fields."""

    @staticmethod
    def validate_hsn(value) -> bool:
        """HSN must be 4,6,8 pure digits (spaces stripped)."""
        if value is None:
            return False
        s = re.sub(r'\s+', '', str(value))
        return s.isdigit() and len(s) in (4, 6, 8)

    @staticmethod
    def validate_item_code(value) -> bool:
        """Item Code must be exactly 7 digits, first digit 1‑4."""
        if value is None:
            return False
        s = str(value)
        return s.isdigit() and len(s) == 7 and s[0] in '1234'

    @staticmethod
    def extract_hsn_from_text(text: str):
        """
        Kept for reference / potential future targeted use (e.g. scanning
        ONLY a confirmed HSN-labeled substring, not the whole item_name).
        NOT called automatically by apply_post_llm_rules() anymore — see
        module docstring for why.
        """
        if not text:
            return None
        matches = re.findall(r'\b(\d{4,8})\b', text)
        for m in matches:
            if len(m) in (4,6,8) and m.isdigit():
                if len(m) != 7:  # avoid confusing with item_code
                    return int(m)
        return None

    @staticmethod
    def compute_total_tax(invoice: dict) -> dict:
        """If Total Tax is missing, sum CGST and SGST from top-level keys."""
        if invoice.get("Total Tax") is not None:
            return invoice
        cgst = invoice.get("CGST")
        sgst = invoice.get("SGST")
        if cgst is not None and sgst is not None:
            invoice["Total Tax"] = cgst + sgst
        elif invoice.get("IGST") is not None:
            invoice["Total Tax"] = invoice["IGST"]
        return invoice

    @staticmethod
    def apply_post_llm_rules(invoices: list) -> list:
        """
        Run this AFTER the LLM returns JSON.
        It will:
          - Nullify invalid HSN/item_code.
          - Compute Total Tax if missing from CGST+SGST/IGST.

        NOTE: no longer attempts a regex HSN-from-item_name fallback here —
        that responsibility now belongs entirely to the Gemma vision
        fallback in llm_extractor.py, which has the header-level-HSN rule
        ("if HSN appears once near the top, apply it to all items") and can
        actually look at the image instead of pattern-matching on
        potentially unrelated numbers embedded in the item description.
        """
        for inv in invoices:
            inv = RuleEngine.compute_total_tax(inv)

            for item in inv.get("Invoice Items", []):
                # Validate HSN
                hsn = item.get("hsn_sac")
                if hsn is not None:
                    if not RuleEngine.validate_hsn(hsn):
                        original = hsn
                        item["hsn_sac"] = None
                        logger.warning(f"Nullified invalid HSN: {original}")

                # HSN cannot equal Item Code
                code = item.get("item_code")
                hsn = item.get("hsn_sac")
                if hsn is not None and code is not None and str(hsn) == str(code):
                    item["hsn_sac"] = None
                    logger.warning(f"HSN equals item_code, nullified HSN for item: {code}")

                # Validate Item Code
                code = item.get("item_code")
                if code is not None:
                    if not RuleEngine.validate_item_code(code):
                        original = code
                        item["item_code"] = None
                        logger.warning(f"Nullified invalid item_code: {original}")

        return invoices