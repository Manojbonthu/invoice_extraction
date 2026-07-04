"""
config.py – Central settings file for the invoice pipeline.

WHAT'S NEW (in simple words):
  1. LOGGING: instead of using print(), every file will now write proper
     log messages (to a file AND to the screen). This helps you find
     problems later, especially in a big batch run.
  2. DEVICE (CPU or GPU): the program checks your computer automatically
     and decides whether to use CPU or GPU. You can also force it by
     writing DEVICE=cpu or DEVICE=gpu in your .env file.
     - If you write DEVICE=gpu but your computer has NO gpu, the program
       will STOP and show a clear error (instead of quietly running slow
       on CPU without telling you).
  3. DEVICE PROFILES: once we know CPU or GPU, we automatically pick good
     default settings (like how many workers, how many pages at once).
     You don't have to remember these numbers yourself.
  4. RATE LIMITER: makes sure we don't send too many requests per minute
     to Gemini/Gemma (which would get blocked by Google).
  5. REAL COST CALCULATION: calculates actual cost per file instead of
     always showing 0.
  6. MODEL CHECK AT STARTUP: tests that your model names are valid BEFORE
     you start processing thousands of files.
"""

import os
import time
import logging
import threading
from collections import deque
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
# STEP 1: LOGGING SETUP
# This replaces every print() in the project. Logs go to both the
# screen (so you can watch it live) and a file (so you can check later).
# ─────────────────────────────────────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = os.getenv("LOG_DIR", "./logs")

_logging_configured = False


def setup_logging() -> logging.Logger:
    """Turns on logging. Safe to call more than once — it only sets up once."""
    global _logging_configured
    root_logger = logging.getLogger()
    if _logging_configured:
        return root_logger

    os.makedirs(LOG_DIR, exist_ok=True)
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    root_logger.setLevel(level)

    log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Writes logs to a file, and automatically starts a new file after 50MB
    # (keeps old files too, up to 10 of them, so disk doesn't fill up forever)
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "pipeline.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)

    # Also prints logs to your terminal screen
    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(log_format)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(screen_handler)
    _logging_configured = True
    return root_logger


setup_logging()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# STEP 2: DEVICE DETECTION (CPU or GPU)
# Rule: if DEVICE is written clearly in .env, always trust that.
# If not written (or written as "auto"), check the computer automatically.
# ─────────────────────────────────────────────────────────

def _gpu_is_available() -> bool:
    """Safely checks if a GPU exists. If torch is missing or broken,
    treat it as 'no GPU' instead of crashing the whole program."""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception as e:
        logger.warning(f"Could not check for GPU (treating as no GPU found): {e}")
        return False


def _resolve_device() -> str:
    """Decides the final answer: 'cpu' or 'gpu'."""
    requested = os.getenv("DEVICE", "auto").strip().lower()

    if requested == "gpu":
        # User explicitly asked for GPU. If none is found, STOP with an
        # error instead of quietly switching to CPU. This protects you
        # from accidentally running a huge batch on CPU by mistake.
        if not _gpu_is_available():
            raise RuntimeError(
                "DEVICE=gpu was set in .env, but no GPU was found on this machine. "
                "Stopping here on purpose, so a slow CPU run doesn't start by accident. "
                "Fix: either run this on a machine with a GPU, or change DEVICE=cpu "
                "(or DEVICE=auto) in your .env file."
            )
        logger.info("Device resolved to: gpu (explicitly set in .env)")
        return "gpu"

    if requested == "cpu":
        # User explicitly asked for CPU. Always honored, no checks needed.
        logger.info("Device resolved to: cpu (explicitly set in .env)")
        return "cpu"

    # requested == "auto" (or anything unrecognized) -> auto-detect quietly
    if _gpu_is_available():
        logger.info("Device resolved to: gpu (auto-detected)")
        return "gpu"
    else:
        logger.info("Device resolved to: cpu (auto-detected, no GPU found)")
        return "cpu"


DEVICE = _resolve_device()


# ─────────────────────────────────────────────────────────
# STEP 3: DEVICE PROFILES
# These are the "good default settings" for each device type.
# Example: on CPU, we use only 1 OCR worker (because CPU has limited
# power and running many at once just slows everything down).
# On GPU, we can use more workers and bigger batches because GPUs are
# built to handle many things at once.
# ─────────────────────────────────────────────────────────

DEVICE_PROFILES = {
    "cpu": {
        "OCR_WORKERS": 1,
        "OCR_BATCH_SIZE": 1,
        "OCR_DPI": 150,
        "MAX_WORKERS": 4,
    },
    "gpu": {
        "OCR_WORKERS": 2,
        "OCR_BATCH_SIZE": 8,
        "OCR_DPI": 200,
        "MAX_WORKERS": 12,
    },
}

_profile = DEVICE_PROFILES[DEVICE]


def _setting(env_name: str, profile_key: str, cast=int):
    """
    Picks the final value for a setting.
    Rule: if you personally wrote it in .env, use that.
    Otherwise, use the automatic profile default for your device.

    Example: if DEVICE=cpu and you didn't set MAX_WORKERS in .env,
    this returns 4 (the CPU default). But if you DID write
    MAX_WORKERS=6 in .env, it returns 6 instead — your choice always wins.
    """
    raw_value = os.getenv(env_name)
    if raw_value is not None:
        return cast(raw_value)
    return _profile[profile_key]


OCR_WORKERS = _setting("OCR_WORKERS", "OCR_WORKERS")
OCR_BATCH_SIZE = _setting("OCR_BATCH_SIZE", "OCR_BATCH_SIZE")
OCR_DPI = _setting("OCR_DPI", "OCR_DPI")
MAX_WORKERS = _setting("MAX_WORKERS", "MAX_WORKERS")

logger.info(
    f"Active settings -> device={DEVICE}, OCR_WORKERS={OCR_WORKERS}, "
    f"OCR_BATCH_SIZE={OCR_BATCH_SIZE}, OCR_DPI={OCR_DPI}, MAX_WORKERS={MAX_WORKERS}"
)


# ─────────────────────────────────────────────────────────
# STEP 4: OTHER SETTINGS (models, folders, keys)
# ─────────────────────────────────────────────────────────

ITEM_CODE_MIN = 1000000
ITEM_CODE_MAX = 4999999

MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16000"))
INPUT_DIR = os.getenv("INPUT_DIR", "./input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./Output_Folder")
DB_PATH = os.getenv("DB_PATH", "./run_status.db")

#GEMMA_VISION_MODEL = os.getenv("GEMMA_VISION_MODEL", "gemma-4-26b-a4b-it")
GEMMA_VISION_MODEL = os.getenv("GEMMA_VISION_MODEL", "gemma-4-31b-it")

GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "60"))  # RPM = requests per minute
GEMMA_RPM_LIMIT = int(os.getenv("GEMMA_RPM_LIMIT", "30"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")


# ─────────────────────────────────────────────────────────
# STEP 5: GEMINI CLIENT (the connection to Google's AI)
# ─────────────────────────────────────────────────────────

_gemini_client = None
_client_lock = threading.Lock()  # prevents two workers from creating 2 clients at once


def get_llm_client():
    """Returns one shared connection to Gemini (created only once)."""
    global _gemini_client
    if _gemini_client is None:
        with _client_lock:
            if _gemini_client is None:
                from google import genai
                _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                logger.info(f"Gemini client ready (model: {MODEL_NAME}, gemma vision: {GEMMA_VISION_MODEL})")
    return _gemini_client


# ─────────────────────────────────────────────────────────
# STEP 6: RATE LIMITER
# This stops us from sending too many requests per minute to Google.
# Think of it like a bucket that can only hold N tokens per minute —
# if it's full, new requests wait their turn instead of crashing.
# ─────────────────────────────────────────────────────────

class TokenBucketLimiter:
    """Keeps requests under a per-minute limit, safely shared across workers."""

    def __init__(self, rpm_limit: int, name: str = "limiter"):
        self.rpm_limit = max(1, rpm_limit)
        self.name = name
        self._recent_calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        """Call this right before making an API request. It may pause
        (sleep) for a moment if we're going too fast."""
        while True:
            with self._lock:
                now = time.monotonic()
                one_minute_ago = now - 60.0
                # forget calls older than 1 minute
                while self._recent_calls and self._recent_calls[0] < one_minute_ago:
                    self._recent_calls.popleft()

                if len(self._recent_calls) < self.rpm_limit:
                    self._recent_calls.append(now)
                    return  # allowed to go now

                wait_time = max(0.05, 60.0 - (now - self._recent_calls[0]))

            logger.debug(f"{self.name}: slowing down for {wait_time:.2f}s to respect rate limit")
            time.sleep(wait_time)


gemini_limiter = TokenBucketLimiter(GEMINI_RPM_LIMIT, name="gemini")
gemma_limiter = TokenBucketLimiter(GEMMA_RPM_LIMIT, name="gemma")


# ─────────────────────────────────────────────────────────
# STEP 7: COST CALCULATION
# Works out the real dollar cost for each API call, based on how many
# tokens (roughly: words/pieces of text) were used.
# IMPORTANT: check Google's actual pricing page and update the numbers
# below before trusting the total cost on a big run.
# ─────────────────────────────────────────────────────────

PRICING_PER_MILLION_TOKENS = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemma-4-26b-a4b-it": {"input": 0.0, "output": 0.0},  # confirm with your plan
    "gemma-4-31b-it": {"input": 0.0, "output": 0.0},      # confirm with your plan
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str = None) -> float:
    """Returns the cost in dollars for one API call."""
    model = model or MODEL_NAME
    prices = PRICING_PER_MILLION_TOKENS.get(model)
    if prices is None:
        logger.warning(f"No price listed for model '{model}' — showing cost as 0.0. "
                        f"Add it to PRICING_PER_MILLION_TOKENS above.")
        return 0.0
    cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    return round(cost, 6)


# ─────────────────────────────────────────────────────────
# STEP 8: STARTUP MODEL CHECK
# Before running thousands of files, test that the model names actually
# work. Better to find out now than after 40,000 files.
# ─────────────────────────────────────────────────────────

def validate_models() -> None:
    """Sends one tiny test message to each model. Stops the program with
    a clear error if any model name is wrong or unavailable."""
    client = get_llm_client()
    for model in (MODEL_NAME, FALLBACK_MODEL, GEMMA_VISION_MODEL):
        try:
            client.models.generate_content(
                model=model,
                contents="ping",
                config={"max_output_tokens": 5},
            )
            logger.info(f"Model check OK: {model}")
        except Exception as e:
            raise RuntimeError(
                f"Could not use model '{model}'. It may not be available on your "
                f"API key/plan — check your Google AI Studio model list. "
                f"Original error: {e}"
            )