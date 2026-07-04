"""
rule_engine.py – Checks and fixes invoice data AFTER the AI (Gemini/Gemma)
returns its answer. Think of this file as a "quality inspector" that
double-checks the AI's work using plain rules (not AI), because rules are
fast, free, and 100% predictable.

WHAT'S NEW (in simple words):
  1. Uses proper logging now (logger) instead of print().
  2. NEW RULE — "Document-level HSN inheritance": some invoices print the
     HSN code only ONCE near the top of the page (like "HSN Code: 8483"),
     instead of writing it next to every single item. Old code missed
     this and left every item's hsn_sac empty. Now we detect that pattern
     and copy the single HSN value to every item that's missing one.
     This is a careful, narrow rule — it only fires when there is exactly
     ONE clearly labeled HSN code found near the header text, not just
     any random 4/6/8-digit number floating in the page (that was the
     old, too-loose rule that caused wrong matches, so we don't bring
     that back).
  3. This new rule runs BEFORE the expensive Gemma vision fallback, so if
     it fixes the missing HSN by itself, we skip paying for an extra AI
     image call. That saves both time and money.
"""

import re
import logging

logger = logging.getLogger(__name__)


class RuleEngine:
    """Plain-rule checks for invoice fields. No AI involved in this file."""

    # A "labeled" HSN mention near the header, e.g. "HSN Code: 8483" or
    # "HSN: 848360". Must have the word HSN/SAC nearby — this is what makes
    # it safe (unlike the old rule, which grabbed ANY stray number).
    _HSN_LABEL_RE = re.compile(
        r'\b(?:HSN|SAC)(?:\s*(?:CODE|NO\.?))?\s*[:\-]?\s*(\d{4,8})\b',
        re.IGNORECASE,
    )

    @staticmethod
    def validate_hsn(value) -> bool:
        """
        HSN/SAC code must be pure digits, and exactly 4, 6, or 8 digits long.
        Example: "8483" is valid (4 digits). "84831" is NOT valid (5 digits).
        """
        if value is None:
            return False
        cleaned = re.sub(r'\s+', '', str(value))
        return cleaned.isdigit() and len(cleaned) in (4, 6, 8)

    @staticmethod
    def validate_item_code(value) -> bool:
        """
        Item Code must be exactly 7 digits, and must start with 1, 2, 3, or 4.
        Example: "1234567" is valid. "5234567" is NOT valid (starts with 5).
        """
        if value is None:
            return False
        cleaned = str(value)
        return cleaned.isdigit() and len(cleaned) == 7 and cleaned[0] in '1234'

    @staticmethod
    def extract_hsn_from_text(text: str):
        """
        Kept only for reference / possible future use. NOT called
        automatically anymore, because it was too loose (it grabbed any
        4/6/8-digit number from the item description, even if it wasn't
        really an HSN code — e.g. a batch number that happened to be
        6 digits long). The safer replacement is
        find_single_document_level_hsn() below, which requires the word
        "HSN" or "SAC" to actually be next to the number.
        """
        if not text:
            return None
        matches = re.findall(r'\b(\d{4,8})\b', text)
        for m in matches:
            if len(m) in (4, 6, 8) and m.isdigit():
                if len(m) != 7:  # avoid confusing with item_code
                    return int(m)
        return None

    @staticmethod
    def find_single_document_level_hsn(texts: list) -> str | None:
        """
        Looks through the invoice's header/footer text lines (NOT the item
        table) for a clearly labeled HSN/SAC code, like "HSN Code: 8483".

        Returns the code ONLY if we find exactly ONE such labeled mention
        in the whole document. If we find zero, or more than one
        (meaning it's ambiguous which one applies), we return None and do
        nothing — better to leave a field empty than guess wrong.

        `texts` is expected to be a list of dicts like
        {"type": "text"/"heading", "text": "..."} — the same shape produced
        by digital_extractor.py and scanned_extractor.py.
        """
        if not texts:
            return None

        found_codes = set()
        for block in texts:
            line = block.get("text", "") if isinstance(block, dict) else str(block)
            match = RuleEngine._HSN_LABEL_RE.search(line)
            if match:
                code = match.group(1)
                if len(code) in (4, 6, 8):
                    found_codes.add(code)

        if len(found_codes) == 1:
            return next(iter(found_codes))

        if len(found_codes) > 1:
            logger.warning(
                f"Found {len(found_codes)} different labeled HSN codes in document "
                f"header text ({found_codes}) — too ambiguous to auto-apply, skipping."
            )
        return None

    @staticmethod
    def apply_document_level_hsn(invoice: dict, texts: list) -> dict:
        """
        If every item in this invoice is missing hsn_sac, and the document
        has exactly one clearly labeled HSN code near the header, copy that
        code into every item. This handles invoice formats that print the
        HSN once for the whole invoice instead of once per item row.

        Runs BEFORE the Gemma vision (image) fallback, so if this rule
        succeeds, we don't need to spend an extra AI call on the image.
        """
        items = invoice.get("Invoice Items", [])
        if not items:
            return invoice

        all_missing = all(item.get("hsn_sac") is None for item in items)
        if not all_missing:
            return invoice  # some items already have it — don't touch anything

        doc_level_hsn = RuleEngine.find_single_document_level_hsn(texts)
        if doc_level_hsn is None:
            return invoice  # nothing safe to apply

        for item in items:
            item["hsn_sac"] = doc_level_hsn
            item["hsn_sac_source"] = "document_level_inheritance"  # helpful for debugging later

        logger.info(
            f"Applied document-level HSN '{doc_level_hsn}' to {len(items)} item(s) "
            f"because it was the only labeled HSN found near the header."
        )
        return invoice

    @staticmethod
    def compute_total_tax(invoice: dict) -> dict:
        """If Total Tax is missing, add it up from CGST+SGST, or use IGST."""
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
    def apply_post_llm_rules(invoices: list, texts: list = None) -> list:
        """
        Run this AFTER the AI returns its JSON answer. In order, it will:
          1. Compute Total Tax if it's missing.
          2. Try the new, safe document-level HSN rule (only if `texts`
             is given — this is the extracted header/footer text lines).
          3. Nullify (blank out) any hsn_sac or item_code that doesn't
             pass the format checks above.
          4. Nullify hsn_sac if it's exactly the same as item_code (they
             should never be equal — that usually means the AI mixed
             them up).

        `texts` is optional so this function still works with old code
        that doesn't pass it — but you should pass it so rule #2 works.
        """
        for inv in invoices:
            inv = RuleEngine.compute_total_tax(inv)

            if texts:
                inv = RuleEngine.apply_document_level_hsn(inv, texts)

            for item in inv.get("Invoice Items", []):
                # Check HSN format
                hsn = item.get("hsn_sac")
                if hsn is not None and not RuleEngine.validate_hsn(hsn):
                    logger.warning(f"Nullified invalid HSN: {hsn}")
                    item["hsn_sac"] = None

                # HSN must not be the same as item_code
                hsn = item.get("hsn_sac")
                code = item.get("item_code")
                if hsn is not None and code is not None and str(hsn) == str(code):
                    logger.warning(f"HSN equals item_code ({code}) — nullified HSN")
                    item["hsn_sac"] = None

                # Check item_code format
                code = item.get("item_code")
                if code is not None and not RuleEngine.validate_item_code(code):
                    logger.warning(f"Nullified invalid item_code: {code}")
                    item["item_code"] = None

        return invoices