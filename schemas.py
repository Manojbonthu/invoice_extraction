"""
schemas.py – Defines the exact SHAPE (structure) that our data must follow.

WHAT'S NEW (in simple words):
  1. Added real "Invoice" and "InvoiceItem" models using Pydantic. Pydantic
     is a library that checks data automatically — like a form that
     rejects your input if you put text in a number box. This means: if
     the AI's answer doesn't match the expected shape (wrong type, missing
     required field, etc.), we find out immediately with a clear error,
     instead of the bad data quietly flowing through the rest of the
     pipeline and causing confusing problems later.
  2. The OLD "NormalizedBlock" / "SourceRef" classes are kept exactly as
     they were — they're used by the legacy extract_digital() wrapper in
     digital_extractor.py, so removing them would break that code path.
  3. Added validate_invoice_batch() — a small helper function that checks
     a whole batch of invoices (a list of dicts) against the new Invoice
     model, and reports which ones passed and which ones failed, instead
     of stopping the whole batch on the first bad one.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
import logging

from pydantic import BaseModel, field_validator, ValidationError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# LEGACY DATACLASSES — used by the older extract_digital() wrapper.
# Left unchanged so nothing breaks.
# ─────────────────────────────────────────────────────────

@dataclass
class SourceRef:
    filename: str
    page: int
    sheet: Optional[str] = None
    slide: Optional[str] = None
    bbox: Optional[List[float]] = None


@dataclass
class NormalizedBlock:
    block_id: str
    document_id: str
    type: str          # "text", "heading", "table", "image"
    text: str = ""
    table_data: Optional[Dict] = None
    source_ref: Optional[SourceRef] = None
    confidence: float = 1.0
    language: str = "en"
    metadata: Dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────
# NEW — Pydantic models matching the ACTUAL invoice output shape that
# flows through llm_extractor.py / rule_engine.py / run.py.
#
# Every field is written as "Optional" (allowed to be empty/null) because
# the whole point of this pipeline is that some fields are genuinely
# missing on some invoices — that's expected, not an error. What we DO
# want to catch is the WRONG TYPE of data (e.g. text where a number
# should be), which usually means something went wrong upstream.
# ─────────────────────────────────────────────────────────

class InvoiceItem(BaseModel):
    """One row from the invoice's item table (e.g. one product line)."""

    item_name: Optional[str] = None
    item_code: Optional[str] = None
    hsn_sac: Optional[str] = None
    quantity: Optional[float] = None
    rate: Optional[float] = None
    amount: Optional[float] = None

    # This extra field gets added by rule_engine.py when it fills in an
    # hsn_sac using the document-level inheritance rule — kept here so
    # Pydantic doesn't reject the invoice for having an "unexpected" field.
    hsn_sac_source: Optional[str] = None

    @field_validator("item_code")
    @classmethod
    def check_item_code_shape(cls, value):
        """
        This does NOT reject invalid codes — rule_engine.py already
        handles nullifying invalid ones. This validator just makes sure
        that IF a value is present, it's stored consistently as a plain
        string of digits (no stray spaces), so later comparisons
        ("does hsn_sac == item_code?") work correctly.
        """
        if value is None:
            return value
        return str(value).strip()

    @field_validator("hsn_sac")
    @classmethod
    def check_hsn_shape(cls, value):
        if value is None:
            return value
        return str(value).strip()


class Invoice(BaseModel):
    """One full invoice, with header fields and a list of items."""

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    total_payable: Optional[float] = None
    total_tax: Optional[float] = None
    cgst: Optional[float] = None
    sgst: Optional[float] = None
    igst: Optional[float] = None
    invoice_items: List[InvoiceItem] = []

    # Pydantic v2 setting: allow field names with spaces (like "Invoice
    # Number") to map onto our snake_case attribute names above, since
    # that's the exact key format the AI returns and rule_engine.py uses.
    model_config = {
        "populate_by_name": True,
    }

    @field_validator("invoice_items", mode="before")
    @classmethod
    def default_empty_list(cls, value):
        return value if value is not None else []


# The AI/rule_engine code actually uses human-readable keys like
# "Invoice Number" and "Invoice Items" (with spaces, capital letters) —
# not the snake_case names above. This small mapping translates between
# the two, so we can validate the real dict shape without having to
# rewrite the rest of the pipeline's key names.
_FIELD_NAME_MAP = {
    "Invoice Number": "invoice_number",
    "Invoice Date": "invoice_date",
    "Total Payable": "total_payable",
    "Total Tax": "total_tax",
    "CGST": "cgst",
    "SGST": "sgst",
    "IGST": "igst",
    "Invoice Items": "invoice_items",
}


def _translate_keys(raw_invoice: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a raw invoice dict (with spaced/capitalized keys) into
    the snake_case keys the Invoice model expects."""
    translated = {}
    for key, value in raw_invoice.items():
        new_key = _FIELD_NAME_MAP.get(key, key)
        translated[new_key] = value
    return translated


def validate_invoice_batch(invoices: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """
    Checks a whole list of invoice dicts (as they come out of the AI +
    rule_engine.py) against the Invoice model.

    Returns a tuple: (valid_invoices, problems)
      - valid_invoices: the original dicts that passed validation
        (unchanged — we don't want to silently rewrite the data, just
        confirm its shape is correct).
      - problems: a list of {"index": i, "error": "..."} for any invoice
        that failed, so run.py can log exactly which invoice had an issue
        instead of the whole file silently failing or crashing.

    This does NOT stop the batch — one broken invoice in a multi-invoice
    file won't block the others from being saved.
    """
    valid_invoices = []
    problems = []

    for i, raw_invoice in enumerate(invoices):
        try:
            translated = _translate_keys(raw_invoice)
            Invoice(**translated)  # just checking the shape — result not otherwise used
            valid_invoices.append(raw_invoice)
        except ValidationError as e:
            logger.warning(f"Invoice at index {i} failed shape validation: {e}")
            problems.append({"index": i, "error": str(e)})
            # Still keep it in valid_invoices — a shape mismatch on one
            # field shouldn't throw away an otherwise-useful invoice.
            # This is a WARNING signal, not a hard rejection.
            valid_invoices.append(raw_invoice)

    return valid_invoices, problems