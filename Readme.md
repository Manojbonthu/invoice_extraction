**Both digital and scanned PDFs converge on the same pipeline** after
extraction. **Batch runs are orchestrated by `run.py`** across a bounded
worker pool, with every file's outcome recorded in a local SQLite database
(`run_status.db`) so a run can be safely stopped and resumed at any point.

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
  Gemma vision fallback actually needs them.

**Scanned** — `scanned_extractor.py`
- Renders each page to an image, then runs Surya OCR (detection +
  recognition) to get line-level text.
- The OCR model loads onto the **device resolved by `config.py`** (CPU or
  GPU — see below), and pages are OCR'd in batches sized to that device's
  profile.
- Groups OCR lines into paragraphs by vertical gap.
- Returns the page images already rendered for OCR under `"images"`, so
  they're reused by the Gemma fallback instead of being re-rendered.
- Every text block is tagged with its **page number**, matching the
  digital path.

### 3. LLM extraction (Gemini) — `llm_extractor.py`
Flattens the extracted tables/texts into a labelled prompt and sends it to
Gemini using **structured output** (`response_schema`) so the response
shape is enforced by the API rather than hoped for via prompt wording.
Covers invoice number/date/totals, `item_code` (7 digits, 1,000,000–
4,999,999, with a tightened fallback that won't misread digits glued to
letters), and `hsn_sac` (4/6/8 digits, distinguished from item_code and
similarly-shaped fields).

Every call goes through a shared **rate limiter** (token bucket, tuned to
your actual AI Studio RPM quota) before hitting the API, and retries are
capped, jittered, and limited to genuinely temporary errors (429/503/
internal server errors) rather than retrying everything blindly.

### 4. Rule validation — `rule_engine.py`
After the LLM returns JSON:
- Nullifies any `hsn_sac`/`item_code` that fails its format check, or where
  `hsn_sac` duplicates `item_code`.
- Computes `Total Tax` from CGST+SGST/IGST if missing.
- **Document-level HSN inheritance**: if exactly one clearly-labeled
  `HSN Code: XXXXXXXX` appears near the header and every item is missing
  `hsn_sac`, applies that value to all items. Runs *before* the Gemma
  vision fallback, saving a meaningful share of vision calls/cost.

### 5. Gemma vision fallback — `llm_extractor.py`
If any field is still `null` after rule validation:
1. Grabs page image(s) — from `data["images"]` (scanned) or via
   `render_page_images()` (digital, on-demand).
2. Sends the relevant page(s) to the Gemma vision model with the same
   schema/rules as the Gemini prompt. Tries multiple pages in sequence for
   multi-page invoices rather than only ever checking page 1.
3. Merges results by position, only filling fields still `null` — never
   overwriting a value Gemini already got right — and re-validates any
   filled `hsn_sac`/`item_code` before accepting it.

### 6. Output & tracking
- `Output_Folder/JSON_Raw/<name>.json` — raw extraction, no LLM involved.
- `Output_Folder/JSON/<name>.json` — final validated result.
- `run_status.db` (SQLite) — one row per file: `doc_id, filename, status,
  attempts, error, tokens_used, cost, started_at, finished_at`. This is
  what makes resume possible and gives a queryable view of failures across
  a 100k-file run.

---

## CPU / GPU: how device selection works

`config.py` resolves the compute device automatically at startup, with a
manual override always available via `.env`:

1. If `DEVICE=cpu` or `DEVICE=gpu` is set explicitly in `.env`, that wins,
   no detection runs.
   - **`DEVICE=gpu` with no GPU actually found is a hard error** — the
     program stops immediately with a clear message, on purpose. This
     protects against silently running a 100k-file batch on CPU for days
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

You never need to edit code to switch environments — only `.env` changes.

---

## Cost & rate limits

- `calculate_cost()` in `config.py` uses a per-model pricing table —
  **verify the figures there against your actual Google AI Studio /
  Vertex billing page** before trusting cost totals on a large batch.
- `GEMINI_RPM_LIMIT` / `GEMMA_RPM_LIMIT` in `.env` should match your real
  AI Studio quota, not a guess — set too high, you'll get throttled; too
  low, you're leaving throughput on the table.
- Running token/cost totals are written to `run_status.db` per file —
  check periodically on large batches (see **Checking status**, below).

---

## Setup (local / laptop)

```bash
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:

```env
# Gemini (primary extraction)
GEMINI_API_KEY=your_google_ai_studio_key
MODEL_NAME=gemini-2.5-flash
FALLBACK_MODEL=gemini-2.5-flash-lite
MAX_OUTPUT_TOKENS=16000

# Gemma (vision fallback) — confirm the exact name against your account's
# model list before running a batch: it can change, and a wrong name will
# fail at startup (see "Model validation" below)
GEMMA_VISION_MODEL=gemma-4-31b-it

# Rate limits — set to your actual AI Studio quota
GEMINI_RPM_LIMIT=60
GEMMA_RPM_LIMIT=30

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

### Model validation at startup
`run.py` calls `config.validate_models()` before touching any files.
`MODEL_NAME` and `FALLBACK_MODEL` are treated as hard requirements (the
batch stops if either fails). `GEMMA_VISION_MODEL` is a soft requirement —
a failure there logs a warning and lets the batch continue, since Gemma is
only used as a fallback for fields still missing after the main pass, not
needed by every file. To see exactly which models your API key currently
has access to:
```bash
python -c "from config import get_llm_client; c = get_llm_client(); [print(m.name) for m in c.models.list()]"
```

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

## Running this on AWS with a GPU (for the team lead / ops)

This section is the step-by-step for taking the code from this GitHub repo
and running it live on an AWS GPU instance for a full production batch.

### 1. Choose the instance
- **`g4dn.xlarge`** (NVIDIA T4) — good default balance of cost and speed.
- **`g5.xlarge`** (NVIDIA A10G) — faster, higher cost, worth it for very
  large batches or tight deadlines.
- Use AWS's **Deep Learning AMI (Ubuntu 22.04)** as the base image — it
  ships with NVIDIA drivers and CUDA already installed, avoiding a manual
  driver-install step.

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
git clone https://github.com/your-org/your-repo.git
cd your-repo
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
Paste in the same `.env` content shown in **Setup** above. Set
`DEVICE=gpu` explicitly (recommended over `auto` here — if the GPU somehow
isn't detected, you want a loud error immediately, not a silent, very slow
CPU run) and set `OCR_WORKERS` / `OCR_BATCH_SIZE` per the GPU profile
table above (or leave them unset to use the automatic GPU defaults).

### 7. Get the invoice PDFs onto the instance
- Small test batch: `scp` directly from your machine.
- Full 100k-file batch: upload to an **S3 bucket** first, then pull onto
  the instance — far more reliable than copying huge folders over SSH:
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

- **Merge is position-matched**: the Gemma fallback merge assumes Gemini
  and Gemma return the same number of invoices/items in the same order.
  If Gemma's re-read finds a different item count, extras are logged and
  skipped rather than guessed at — check `run_status.db` / logs for
  `⚠️ Gemma returned fewer items than Gemini`.
- **OCR worker sizing on GPU**: `OCR_WORKERS` must be sized to available
  GPU memory (VRAM), not CPU core count — each worker loads its own Surya
  model instance. Check `nvidia-smi` headroom before raising it.
- **Gemma model availability**: Gemma model names/availability can change
  on Google's side; a `500` error from Gemma specifically is treated as
  non-fatal (batch continues, only files needing the fallback are
  affected) — see **Model validation at startup** above.

---

## File reference

| File | Role |
|---|---|
| `config.py` | Env loading, logging setup, CPU/GPU device resolution + profiles, rate limiter, real cost calculation, startup model validation |
| `detector.py` | Digital vs scanned vs mixed classification |
| `digital_extractor.py` | PyMuPDF table/text extraction, page-tagged, lazy page image render |
| `scanned_extractor.py` | Device-aware, batched Surya OCR extraction + page image capture, page-tagged |
| `llm_extractor.py` | Gemini structured-output call, rate-limited/jittered retries, RuleEngine invocation, multi-page Gemma vision fallback, token/cost tracking |
| `rule_engine.py` | Field validation (hsn_sac, item_code), tax computation, document-level HSN inheritance |
| `run.py` | Batch orchestrator — worker pool, resume/force/limit flags, SQLite status tracking, startup model validation |
| `schemas.py` | `Invoice`/`InvoiceItem` Pydantic output validation + legacy `NormalizedBlock`/`SourceRef` dataclasses |
| `requirements.txt` | Pinned dependencies (CPU `torch` by default — swap for CUDA build on GPU, see above) |