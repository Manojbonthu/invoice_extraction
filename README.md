# Invoice Extraction Pipeline

Extracts structured invoice data (invoice number, date, totals, and per-item
fields including HSN/SAC codes and item codes) from both **digital** (text-based)
and **scanned** (image-based) PDF invoices, using Gemini for the primary
extraction and a **Gemma vision fallback** to recover fields that are still
`null` after the first pass.

---

## How it works

```
                          ┌─────────────────┐
                          │   detector.py    │
                          │ digital/scanned? │
                          └────────┬─────────┘
                                   │
                 ┌─────────────────┴─────────────────┐
                 │                                     │
        ┌────────▼─────────┐               ┌───────────▼──────────┐
        │ digital_extractor │               │  scanned_extractor    │
        │  (PyMuPDF tables   │               │  (Surya OCR, plain     │
        │   + text)          │               │   text + page images)  │
        └────────┬───────────┘               └───────────┬────────────┘
                 │                                         │
                 └───────────────┬─────────────────────────┘
                                  │  same dict shape:
                                  │  {"tables": [...], "texts": [...],
                                  │   "images": [...] (scanned only)}
                                  ▼
                       ┌─────────────────────┐
                       │   llm_extractor.py    │
                       │  1. Gemini text pass   │
                       │  2. RuleEngine checks  │
                       │  3. Gemma vision       │
                       │     fallback if any    │
                       │     field still null   │
                       └──────────┬─────────────┘
                                  │
                                  ▼
                       Output_Folder/JSON/<name>.json
```

**Both digital and scanned PDFs converge on the same pipeline** after
extraction — they produce the same `{"tables": [...], "texts": [...]}` dict
shape, so `llm_extractor.py` doesn't need to know which type of PDF it came
from.

---

## Pipeline stages

### 1. Detection — `detector.py`
Opens the PDF with PyMuPDF and checks extractable text per page. A page with
more than 5 characters of extractable text is `digital`; otherwise `scanned`.
A PDF is `mixed` if pages differ.

### 2. Extraction

**Digital** — `digital_extractor.py`
- Uses PyMuPDF's `find_tables()` for structured tables (math-based density
  filtering, no hardcoded keywords).
- Extracts all remaining text blocks (headers, footers, metadata) without
  skipping content that overlaps detected tables.
- `render_page_images()` — lazily renders page(s) to PNG only when the Gemma
  vision fallback actually needs them (digital PDFs rarely do).

**Scanned** — `scanned_extractor.py`
- Renders each page to an image (`pdf_to_images()`), then runs Surya OCR
  (detection + recognition) to get line-level text with bounding boxes.
- Groups OCR lines into paragraphs by vertical gap — no geometric table
  reconstruction (found to be less reliable than a strong LLM prompt for this
  invoice format).
- Returns the **same page images already rendered for OCR** under `"images"`
  in the result dict, so they're reused by the Gemma fallback instead of
  being re-rendered.

### 3. LLM extraction (Gemini) — `llm_extractor.py`
`format_data_for_llm()` flattens the extracted tables/texts into a labelled
plain-text prompt (`=== INVOICE HEADER ===`, `=== Table N: ITEM TABLE ===`,
etc.) and sends it to Gemini with a detailed system prompt covering:
- Invoice number / date / totals extraction rules.
- `item_code`: 7 digits (1,000,000–4,999,999), with a defined label priority
  (F.G Code > Input Code) and a description-scan fallback.
- `hsn_sac`: 4/6/8 digits, distinguished from item_code and other
  similarly-shaped fields (Excise No, EWB No, GST No).

`call_api()` retries with exponential backoff on 429/503, falling back from
`MODEL_NAME` to `FALLBACK_MODEL` after 3 attempts.

### 4. Rule validation — `rule_engine.py`
After the LLM returns JSON:
- Nullifies any `hsn_sac` that isn't exactly 4/6/8 digits.
- Nullifies any `item_code` that isn't exactly 7 digits starting with 1–4.
- Nullifies `hsn_sac` if it duplicates `item_code`.
- Computes `Total Tax` from CGST+SGST/IGST if the LLM didn't return it directly.
- Attempts a regex-based HSN fallback from `item_name` if still null.

### 5. Gemma vision fallback — `llm_extractor.py`
If **any** field (invoice-level or item-level) is still `null` after step 4:
1. Grabs the page image — from `data["images"]` (scanned) or via
   `render_page_images()` (digital, rendered on-demand).
2. Sends the **full page image** to a Gemma vision model with the **same
   schema and extraction rules** as the Gemini prompt (not just the missing
   fields — a full re-extraction, for a more reliable/consistent read).
3. Merges the result: for each invoice/item, matched by list position,
   **only fills fields that are still `null`** — never overwrites a value
   Gemini already got right.
4. Re-validates any filled `hsn_sac`/`item_code` through
   `RuleEngine.validate_hsn()` / `validate_item_code()` before accepting it.

### 6. Output
- `Output_Folder/JSON_Raw/<name>.json` — raw extraction (tables/texts, no
  images, no LLM involvement) — useful for re-running the LLM/Gemma steps
  without re-doing OCR (`test_gemma_fallback.py` uses this).
- `Output_Folder/JSON/<name>.json` — final validated result.

---

## Setup

### Requirements
```
pip install -r requirements.txt
```

### `.env`
```env
# Gemini (primary extraction)
GEMINI_API_KEY=your_google_ai_studio_key
MODEL_NAME=gemini-2.5-flash
FALLBACK_MODEL=gemini-2.5-flash-lite
MAX_OUTPUT_TOKENS=16000
REQUEST_DELAY=0.3

# Gemma (vision fallback for null fields)
GEMMA_VISION_MODEL=gemma-4-26b-a4b-it

# Input
INPUT_DIR=./input
```

**Notes on `GEMMA_VISION_MODEL`:**
- Must match a model actually available on your API key/tier. Check your
  Google AI Studio model list before assuming a model string — Gemma's
  lineup has moved fast (Gemma 3 → Gemma 4), and a hardcoded string can
  404 if your account's available models differ.
- `gemma-4-26b-a4b-it` (Mixture-of-Experts, ~4B active params) is the
  faster/cheaper option — good default for this task since it's a narrow
  field-extraction job, not deep reasoning. `gemma-4-31b-it` (dense, 256K
  context) is the higher-quality alternative if the smaller model makes
  mistakes on your invoices.
- Both `GEMINI_API_KEY` and Gemma calls go through the same
  `google-genai` client / API key — no separate credential needed.
- Check your account's rate limit for the Gemma model (shown in AI Studio)
  before running large batches — free tiers are often much lower RPM than
  Gemini's.

---

## Usage

### Run the full pipeline
```bash
# Process every PDF found under INPUT_DIR/1.Input/
python run.py

# Process a single PDF
python run.py "PDF\2_Scanned.pdf"
```

### Test the Gemma fallback in isolation (no re-extraction)
Useful for iterating on the Gemma prompt without waiting through OCR again.
```bash
python test_gemma_fallback.py "Output_Folder\JSON_Raw\2_Scanned.json" "PDF\2_Scanned.pdf"
```
This loads the already-saved raw extraction, re-runs the Gemini + RuleEngine
+ Gemma fallback chain, prints null counts before/after, and saves to
`Output_Folder/JSON_TestFallback/`.

---

## Known limitations / things to watch

- **Single-page assumption**: the Gemma fallback currently sends only
  `images[0]` to Gemma. Fine for this invoice set (almost all single-page),
  but will need extending (loop pages, or track which page an item came
  from) if multi-page invoices with items split across pages become common.
- **Document-level HSN codes**: some invoice formats (e.g. Vishal Bearings)
  print a single `HSN Code: XXXXXXXX` once near the header rather than per
  item row. The extraction prompts need an explicit rule telling the model
  to apply that single value to all items in the invoice — without it, both
  Gemini and Gemma will correctly report "not found per item" and leave
  `hsn_sac` null even though the value is visible on the page.
- **item_code false positives**: the "scan item description for any 7-digit
  number starting with 1-4" fallback rule can grab non-item-code numbers
  embedded in batch/reference strings (e.g. `502==F-2500828` → wrongly
  extracted as `2500828`). Worth tightening (e.g. requiring the number not
  be immediately preceded by letters) or removing if it causes more false
  positives than genuine saves.
- **Merge is position-matched**: the Gemma fallback merge assumes Gemini and
  Gemma return the same number of invoices/items in the same order. If
  Gemma's re-read finds a different item count, extras are logged and
  skipped rather than guessed at — check logs for
  `⚠️ Gemma returned fewer items than Gemini` if this happens.
- **`test_pdf.py`** currently imports `extract_digital_text_only` /
  `extract_scanned_text_only`, which no longer match the current dict-based
  return shape of `digital_extractor.py` — needs updating separately from
  the fallback feature (currently broken if run as-is).

---

## File reference

| File | Role |
|---|---|
| `config.py` | Env loading, Gemini client singleton, model name constants |
| `detector.py` | Digital vs scanned vs mixed classification |
| `digital_extractor.py` | PyMuPDF table/text extraction + lazy page image render |
| `scanned_extractor.py` | Surya OCR extraction + page image capture |
| `llm_extractor.py` | Gemini prompt/call, RuleEngine invocation, Gemma vision fallback |
| `rule_engine.py` | Field validation (hsn_sac, item_code), tax computation, HSN regex fallback |
| `run.py` | Orchestrates the full per-PDF pipeline, batch or single-file |
| `test_gemma_fallback.py` | Standalone test: raw JSON → LLM → Gemma fallback, no re-extraction |
| `test_pdf.py` | ⚠️ Currently broken — imports functions removed from `digital_extractor.py` |
| `schemas.py` | `NormalizedBlock`/`SourceRef` dataclasses (legacy extractor wrapper) |
| `utils.py` | File/text/date helpers, retry decorator |
