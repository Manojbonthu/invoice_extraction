# Invoice Extraction Pipeline

A production-grade pipeline for extracting structured invoice data —
invoice number, date, totals, and per-item fields including HSN/SAC codes
and item codes — from both **digital** (text-based) and **scanned**
(image-based) PDF invoices.

Text extraction is handled by an LLM (**Gemini or OpenAI**, switchable
with one setting — see **LLM provider** below), a **rule engine**
validates the result deterministically, and a **vision-model fallback**
recovers fields that are still `null` after the first pass by reading the
actual page image.

**Both digital and scanned PDFs converge on the same pipeline** after
extraction. **Batch runs are orchestrated by `run.py`** across a bounded
worker pool, with every file's outcome recorded in a local SQLite database
(`run_status.db`) so a run can be safely stopped and resumed at any point.

---

## LLM provider: Gemini or OpenAI, one setting

`config.py` supports two interchangeable LLM providers, picked by a single
`.env` value:

```env
LLM_PROVIDER=gemini   # or: openai
```

Everything downstream (`llm_extractor.py`, `run.py`, `rule_engine.py`)
only ever refers to three **generic** model names —
`config.PRIMARY_MODEL`, `config.FALLBACK_MODEL`, `config.VISION_MODEL` —
which `config.py` resolves to the right provider's actual model names.
No other file in the codebase needs to change when you switch providers.

| | Gemini | OpenAI |
|---|---|---|
| Primary extraction | `gemini-2.5-flash` | `gpt-4o-mini` (or your choice) |
| Fallback (after repeated primary failures) | `gemini-2.5-flash-lite` | same as primary by default |
| Vision fallback (reads the page image) | `gemma-4-31b-it` (separate model) | same model as primary — GPT-4o family reads images natively |
| API key required | `GEMINI_API_KEY` | `OPENAI_API_KEY` |

Only the **active** provider's API key is required at startup — running
`LLM_PROVIDER=gemini` locally does not require `OPENAI_API_KEY` to be set,
and vice versa. This is the intended workflow: **develop against Gemini
locally, deploy against OpenAI in production**, by changing one line in
`.env` — no code changes.

To actually decide which provider extracts *your* invoices more
accurately: run the same batch of 50–100 files through both (flip
`LLM_PROVIDER`, rerun with `--force`), and compare null-rates on
`hsn_sac`/`item_code` in the resulting JSON. Don't take either provider's
reputation on faith — invoice formats vary too much for a generic answer.

---

## Pipeline stages

### 1. Detection — `detector.py`
Opens the PDF with PyMuPDF and checks extractable text per page. A page
with more than 5 characters of extractable text is `digital`; otherwise
`scanned`. A PDF is `mixed` if pages differ.

### 2. Extraction

**Digital** — `digital_extractor.py`
- Uses PyMuPDF's `find_tables()` for structured tables (math-based density
  filtering, no hardcoded keywords).
- Extracts all remaining text blocks without skipping content that
  overlaps detected tables.
- Every table and text block is tagged with its **page number**, so
  multi-page invoices can be matched correctly downstream.
- `render_page_images()` lazily renders page(s) to PNG only when the
  vision fallback actually needs them. Page count for the final log line
  is captured *before* the document is closed (a `RuntimeError: document
  closed` bug from calling `len(doc)` after `doc.close()` was fixed here).

**Scanned** — `scanned_extractor.py`
- Renders each page to an image, then runs Surya OCR (detection +
  recognition) to get line-level text.
- The OCR model loads onto the **device resolved by `config.py`** (CPU or
  GPU — see below).
- **Two ways to run OCR, depending on scale:**
  - `extract_scanned_clean()` / `extract_scanned_text_only()` — single-PDF
    OCR, still used by test/one-off scripts. Batches pages *within* one
    PDF according to `OCR_BATCH_SIZE`.
  - `create_pdf_aware_batches()` + `extract_scanned_batch()` — used by
    `run.py` for real batch runs. Groups **multiple PDFs together** by
    total page count (never splitting one PDF's pages across two groups)
    and OCRs an entire group in **one model call**. This is what keeps
    GPU usage to one large, orderly batch at a time instead of several
    worker threads separately hammering the same GPU-resident model.
- Groups OCR lines into paragraphs by vertical gap.
- Returns the page images already rendered for OCR under `"images"`, so
  they're reused by the vision fallback instead of being re-rendered.
- Every text block is tagged with its **page number**, matching the
  digital path.

### 3. LLM extraction — `llm_extractor.py`
Flattens the extracted tables/texts into a labelled prompt and sends it to
the active provider's primary model using **structured output**
(`response_schema` for Gemini, strict `json_schema` mode for OpenAI) so
the response shape is enforced by the API rather than hoped for via
prompt wording. Covers invoice number/date/totals, `item_code` (7 digits,
1,000,000–4,999,999, with a tightened fallback that won't misread digits
glued to letters), and `hsn_sac` (4/6/8 digits, distinguished from
item_code and similarly-shaped fields).

Every call goes through a shared **rate limiter** (token bucket, tuned to
your actual provider quota) before hitting the API, and retries are
capped, jittered, and limited to genuinely temporary errors (429/500/502/
503, timeouts) rather than retrying everything blindly. Usage/cost per
call is returned directly from `call_api()` and accumulated **locally**
per file — not a shared global — so it stays correct under concurrent
worker threads.

### 4. Rule validation — `rule_engine.py`
After the LLM returns JSON:
- Nullifies any `hsn_sac`/`item_code` that fails its format check, or
  where `hsn_sac` duplicates `item_code`.
- Computes `Total Tax` from CGST+SGST/IGST if missing.
- **Document-level HSN inheritance**: if exactly one clearly-labeled
  `HSN Code: XXXXXXXX` appears near the header and every item is missing
  `hsn_sac`, applies that value to all items. Runs *before* the vision
  fallback, saving a meaningful share of vision-model calls/cost.

### 5. Vision fallback — `llm_extractor.py`
If any field is still `null` after rule validation:
1. Grabs page image(s) — from `data["images"]` (scanned) or via
   `render_page_images()` (digital, on-demand).
2. Sends the relevant page(s) to `config.VISION_MODEL` (Gemma or GPT-4o
   family, depending on `LLM_PROVIDER`) with the same schema/rules as the
   primary prompt. Tries multiple pages in sequence for multi-page
   invoices rather than only ever checking page 1.
3. Merges results by position, only filling fields still `null` — never
   overwriting a value the primary pass already got right — and
   re-validates any filled `hsn_sac`/`item_code` before accepting it.

### 6. Output & tracking
- `Output_Folder/JSON_Raw/<name>.json` — raw extraction, no LLM involved.
- `Output_Folder/JSON/<name>.json` — final validated result.
- `run_status.db` (SQLite) — one row per file: `doc_id, filename, status,
  attempts, error, tokens_used, cost, started_at, finished_at`. This is
  what makes resume possible and gives a queryable view of failures across
  a large batch.

  **Note:** `print_summary()` at the end of a run reports **lifetime
  totals across every run ever recorded in `run_status.db`**, not just
  the files touched in the run that just finished. If you rerun `run.py`
  on a file already marked `success`, it's skipped entirely (see
  **Usage** below) and the summary you see afterward still reflects
  everything the database has ever recorded — don't mistake it for
  "what did this specific run just do."

---

## CPU / GPU: how device selection works

`config.py` resolves the compute device automatically at startup, with a
manual override always available via `.env`:

1. If `DEVICE=cpu` or `DEVICE=gpu` is set explicitly in `.env`, that wins,
   no detection runs.
   - **`DEVICE=gpu` with no GPU actually found is a hard error** — the
     program stops immediately with a clear message, on purpose. This
     protects against silently running a large batch on CPU for days
     because a `.env` file was copied from the wrong machine.
2. If `DEVICE` is unset or `auto`, it checks `torch.cuda.is_available()`
   and picks `gpu` or `cpu` accordingly, logging the result.

Once the device is known, a **profile** supplies sane defaults for
`OCR_WORKERS`, `OCR_BATCH_SIZE`, `OCR_DPI`, and `MAX_WORKERS` — any of
which can still be overridden individually in `.env` without losing the
rest of the profile's defaults.

| Setting | CPU profile | GPU profile |
|---|---|---|
| `OCR_WORKERS` | 1 | 2 |
| `OCR_BATCH_SIZE` | 1 | 8 |
| `OCR_DPI` | 150 | 200 |
| `MAX_WORKERS` | 4 | 12 |

`OCR_BATCH_SIZE` is a **page-count budget**, used two ways depending on
which OCR path runs: as the max pages sent in one call within a single
PDF (`extract_scanned_clean`), and as the max total pages grouped across
multiple PDFs in one batch (`create_pdf_aware_batches`, used by `run.py`).

Note this device setting only affects **OCR** (Surya) — it's completely
separate from `LLM_PROVIDER`. You can run OCR on GPU while using either
Gemini or OpenAI for the LLM calls; they don't interact.

You never need to edit code to switch environments — only `.env` changes.

---

## Cost & rate limits

- `calculate_cost()` in `config.py` uses a per-model pricing table
  covering both providers — **verify the figures there against the
  provider's current pricing page** (Google AI Studio/Vertex, or
  platform.openai.com/pricing) before trusting cost totals on a large
  batch. Prices change; the table in code is a snapshot, not a live feed.
- `GEMINI_RPM_LIMIT` / `GEMMA_RPM_LIMIT` (Gemini) or `OPENAI_RPM_LIMIT`
  (OpenAI) in `.env` should match your real account quota, not a guess —
  set too high, you'll get throttled; too low, you're leaving throughput
  on the table. Only the active provider's limit(s) actually apply.
- Your rate limit is a hard ceiling on throughput regardless of worker
  count. Rough math before a big batch: `file_count ÷ RPM_limit` gives you
  a *minimum* wall-clock time, before counting any vision-fallback calls
  or retries on top.
- Running token/cost totals are written to `run_status.db` per file —
  check periodically on large batches (see **Checking status**, below).

---

## Setup (local / laptop)

```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root. Pick **one** of the two provider blocks
below (only the active provider's key is required):

```env
# ── LLM provider switch ──────────────────────────────────
LLM_PROVIDER=gemini   # or: openai

# ── Gemini settings (used if LLM_PROVIDER=gemini) ────────
GEMINI_API_KEY=your_google_ai_studio_key
MODEL_NAME=gemini-2.5-flash
FALLBACK_MODEL=gemini-2.5-flash-lite
GEMMA_VISION_MODEL=gemma-4-31b-it
GEMINI_RPM_LIMIT=60
GEMMA_RPM_LIMIT=30

# ── OpenAI settings (used if LLM_PROVIDER=openai) ────────
# OPENAI_API_KEY=your_openai_key
# OPENAI_MODEL_NAME=gpt-4o-mini
# OPENAI_FALLBACK_MODEL=gpt-4o-mini
# OPENAI_VISION_MODEL=gpt-4o-mini
# OPENAI_RPM_LIMIT=500

# ── Shared settings ───────────────────────────────────────
MAX_OUTPUT_TOKENS=16000

# Device — "auto" detects CPU/GPU automatically; set explicitly to force one
DEVICE=auto

# Concurrency / OCR (optional — auto-filled from the device profile above
# if left unset; override individually here only if you need to)
# MAX_WORKERS=4
# OCR_WORKERS=1
# OCR_BATCH_SIZE=1
# OCR_DPI=150

# Input / output / tracking
INPUT_DIR=./input
OUTPUT_DIR=./Output_Folder
DB_PATH=./run_status.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=./logs
```

**Confirm your exact model names before running a batch** — especially
`GEMMA_VISION_MODEL`, which can change on Google's side, or whichever
OpenAI model you pick — a wrong name fails loudly at startup (see below),
not silently mid-batch.

### Model validation at startup
`run.py` calls `config.validate_models()` before touching any files,
pinging `PRIMARY_MODEL`, `FALLBACK_MODEL`, and `VISION_MODEL` for
whichever provider is active. A failure here stops the batch immediately
with a clear error, rather than failing thousands of files in on the
first vision-fallback call.

To see exactly which Gemini models your API key currently has access to:
```bash
python -c "from config import get_llm_client; c = get_llm_client(); [print(m.name) for m in c.models.list()]"
```
(For OpenAI, check your available models at platform.openai.com.)

---

## Usage

```bash
# Process every PDF under INPUT_DIR/1.Input/, skipping already-completed files
python run.py --workers 8

# Force reprocessing of everything, including already-completed files
python run.py --workers 8 --force

# Quick test on a handful of files before a big run
python run.py --workers 8 --limit 20

# Process a single PDF
python run.py "PDF-4/49.pdf"
```

**Before pointing this at a large batch (thousands of files), stage it:**
run `--limit 20`, then `--limit 200`, then `--limit 2000`, checking
`run_status.db` for failures and the running cost between each stage,
rather than jumping straight to the full batch. This catches format-
specific extraction issues, quota surprises, or GPU memory limits on a
small run instead of hours into a large one.

### Checking status
```bash
sqlite3 run_status.db
.headers on
.mode column
SELECT status, COUNT(*), SUM(tokens_used), SUM(cost) FROM file_status GROUP BY status;
SELECT filename, error FROM file_status WHERE status = 'failed';
```
(Or use a GUI tool like [DB Browser for SQLite](https://sqlitebrowser.org/)
if you prefer not to use the command line.)

---

## Running this on AWS with a GPU

This section is the step-by-step for taking the code from this GitHub repo
and running it live on an AWS GPU instance for a full production batch.
This is entirely about OCR hardware — it applies the same way regardless
of which `LLM_PROVIDER` you're using.

### 1. Choose the instance
- **`g4dn.xlarge`** (NVIDIA T4) — good default balance of cost and speed.
- **`g5.xlarge`** (NVIDIA A10G) — faster, higher cost, worth it for very
  large batches.

### 2. Launch & connect
- Security group: allow inbound **SSH (port 22)** from your IP only — this
  pipeline is a batch script, not a web service, so no other ports are
  needed.
- Connect:
```bash
ssh -i your-key.pem ubuntu@<instance-public-ip>
```

### 3. Get the code
```bash
git clone https://github.com/Manojbonthu/invoice_extraction.git
cd invoice_extraction
```
(For a private repo, set up a GitHub personal access token or deploy key
on the instance first.)

### 4. Verify Python version
```bash
python3 --version
```
If it isn't 3.11.x:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### 5. Set up the environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**GPU-specific step:** `requirements.txt` pins the CPU build of `torch` by
default so it installs anywhere with zero setup. On this GPU instance,
replace it with the CUDA build matching the instance's CUDA version
(check with `nvidia-smi` first):
```bash
pip uninstall torch
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### 6. Add secrets
`.env` is git-ignored on purpose (never commit API keys) — create it
directly on the instance:
```bash
nano .env
```
Paste in the same `.env` content shown in **Setup** above — with
`LLM_PROVIDER` and the matching key(s) set for whichever provider this
environment should use. Set `DEVICE=gpu` explicitly (recommended over
`auto` here — if the GPU somehow isn't detected, you want a loud error
immediately, not a silent, very slow CPU run) and set `OCR_WORKERS` /
`OCR_BATCH_SIZE` per the GPU profile table above (or leave them unset to
use the automatic GPU defaults).

### 7. Get the invoice PDFs onto the instance
- Small test batch: `scp` directly from your machine.
- Full large batch: upload to an **S3 bucket** first, then pull onto the
  instance — far more reliable than copying huge folders over SSH:
```bash
  aws s3 sync s3://your-bucket/invoices ./input/1.Input
```

### 8. Test small before running the full batch
```bash
python run.py --limit 20
```
Check `logs/pipeline.log`, `Output_Folder/JSON/`, and `run_status.db`
before scaling up.

### 9. Run the full batch — and keep it alive after disconnecting
A plain `python run.py` dies if the SSH session drops. Use `tmux`:
```bash
tmux new -s invoice_batch
python run.py --workers 12
```
Detach with `Ctrl+B` then `D` — the batch keeps running in the background.
Reattach anytime with:
```bash
tmux attach -t invoice_batch
```

### 10. Monitor remotely
```bash
tail -f logs/pipeline.log
```
Or check `run_status.db` as shown in **Checking status** above, from a
second terminal session, at any time — including while the batch is still
running.

---

## Known limitations / things to watch

- **No per-request timeout on the LLM API call.** A single hung request
  (network stall, provider-side issue) can occupy a worker thread
  indefinitely, and since `ThreadPoolExecutor` waits for every submitted
  task before its `with` block exits, one stuck file can delay the entire
  batch finishing. Not yet fixed — worth adding an explicit timeout before
  a very large unattended run.
- **No circuit breaker on repeated failures.** If your API quota runs out
  mid-batch, every remaining file will still burn through its full retry
  budget (up to `max_retries=5` with growing backoff) before failing,
  rather than detecting "this looks like quota exhaustion" and stopping
  early. On a large batch this can waste hours. Not yet fixed.
- **Merge is position-matched**: the vision fallback merge assumes the
  primary pass and the vision pass return the same number of
  invoices/items in the same order. If the vision re-read finds a
  different item count, extras are logged and skipped rather than guessed
  at — check `run_status.db` / logs for
  `⚠️ Vision fallback returned fewer items than the primary pass`.
- **OCR worker sizing on GPU**: `OCR_WORKERS` and `OCR_BATCH_SIZE` must be
  sized to available GPU memory (VRAM), not guessed from the default
  profile — the cross-PDF OCR batching path is new and hasn't been
  validated against real GPU hardware at scale yet. Test on a real GPU
  instance with a moderate batch (20-30 scanned files) before trusting it
  on thousands.
- **Vision model availability**: exact model names/availability can change
  on either provider's side; confirm yours against your account before a
  batch (see **Model validation at startup** above).
- **`print_summary()` reports lifetime totals**, not per-run totals — see
  the note under **Output & tracking** above.

---

## File reference

| File | Role |
|---|---|
| `config.py` | Env loading, logging setup, CPU/GPU device resolution + profiles, **LLM provider switch (Gemini/OpenAI)**, rate limiter, real cost calculation, startup model validation |
| `detector.py` | Digital vs scanned vs mixed classification |
| `digital_extractor.py` | PyMuPDF table/text extraction, page-tagged, lazy page image render |
| `scanned_extractor.py` | Device-aware, batched Surya OCR extraction (single-PDF and cross-PDF page-budget batching) + page image capture, page-tagged |
| `llm_extractor.py` | Provider-agnostic dispatcher (Gemini/OpenAI backends), structured-output calls, rate-limited/jittered retries, RuleEngine invocation, multi-page vision fallback, per-file token/cost tracking |
| `rule_engine.py` | Field validation (hsn_sac, item_code), tax computation, document-level HSN inheritance |
| `run.py` | Batch orchestrator — worker pool for digital PDFs, OCR-batch grouping for scanned PDFs, resume/force/limit flags, SQLite status tracking, startup model validation |
| `schemas.py` | `Invoice`/`InvoiceItem` Pydantic output validation + legacy `NormalizedBlock`/`SourceRef` dataclasses |
| `requirements.txt` | Pinned dependencies (CPU `torch` by default — swap for CUDA build on GPU, see above; add `openai>=1.40.0` if using `LLM_PROVIDER=openai`) |