"""
config.py – Central settings file for the invoice pipeline.

PROVIDER ABSTRACTION (NEW):
  This file now supports TWO LLM providers — Google Gemini and OpenAI —
  switchable with ONE setting: LLM_PROVIDER in your .env file.
    LLM_PROVIDER=gemini   -> uses Gemini (primary) + Gemma (vision fallback)
    LLM_PROVIDER=openai   -> uses GPT-4o family for both primary + vision

  Everything else in the pipeline (llm_extractor.py, run.py, rule_engine.py)
  talks to THREE generic names instead of provider-specific ones:
    config.PRIMARY_MODEL   - the main text-extraction model
    config.FALLBACK_MODEL  - used after repeated primary-model failures
    config.VISION_MODEL    - used for the page-image fallback pass
  These automatically point at the right model names for whichever
  provider is active, so switching providers never requires touching
  llm_extractor.py.

  Only the API key for the ACTIVE provider is required at startup — if
  you're running LLM_PROVIDER=gemini locally, you do NOT need an
  OPENAI_API_KEY set, and vice versa in production. This is the whole
  point: you can develop against Gemini locally and deploy against
  OpenAI in prod by changing ONE line in .env, no code changes.
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
# STEP 1: LOGGING SETUP (unchanged)
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

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, "pipeline.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(log_format)

    screen_handler = logging.StreamHandler()
    screen_handler.setFormatter(log_format)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(screen_handler)
    _logging_configured = True
    return root_logger


setup_logging()
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# STEP 2: DEVICE DETECTION (CPU or GPU) — unchanged.
# This is about OCR hardware, completely separate from which LLM
# provider you use for text extraction.
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
        logger.info("Device resolved to: cpu (explicitly set in .env)")
        return "cpu"

    if _gpu_is_available():
        logger.info("Device resolved to: gpu (auto-detected)")
        return "gpu"
    else:
        logger.info("Device resolved to: cpu (auto-detected, no GPU found)")
        return "cpu"


DEVICE = _resolve_device()


# ─────────────────────────────────────────────────────────
# STEP 3: DEVICE PROFILES (OCR settings) — unchanged
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
    """If you wrote it in .env, that wins. Otherwise use the device's
    profile default."""
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
# STEP 4: OTHER SETTINGS (folders, item-code range) — unchanged
# ─────────────────────────────────────────────────────────

ITEM_CODE_MIN = 1000000
ITEM_CODE_MAX = 4999999

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16000"))
INPUT_DIR = os.getenv("INPUT_DIR", "./input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./Output_Folder")
DB_PATH = os.getenv("DB_PATH", "./run_status.db")


# ─────────────────────────────────────────────────────────
# STEP 5: LLM PROVIDER SWITCH (NEW)
# This ONE setting decides everything below it: which SDK gets used,
# which API key is required, which models get called, which rate
# limiter applies.
# ─────────────────────────────────────────────────────────

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
if LLM_PROVIDER not in ("gemini", "openai"):
    raise ValueError(
        f"LLM_PROVIDER must be 'gemini' or 'openai', got '{LLM_PROVIDER}'. "
        f"Set it in your .env file."
    )

# --- Gemini-specific settings (only enforced if LLM_PROVIDER=gemini) ---
GEMINI_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gemini-2.5-flash-lite")
GEMMA_VISION_MODEL = os.getenv("GEMMA_VISION_MODEL", "gemma-4-31b-it")
GEMINI_RPM_LIMIT = int(os.getenv("GEMINI_RPM_LIMIT", "60"))
GEMMA_RPM_LIMIT = int(os.getenv("GEMMA_RPM_LIMIT", "30"))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- OpenAI-specific settings (only enforced if LLM_PROVIDER=openai) ---
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_FALLBACK_MODEL = os.getenv("OPENAI_FALLBACK_MODEL", "gpt-4o-mini")
# GPT-4o and GPT-4o-mini both read images natively — default the vision
# fallback to the SAME model unless you deliberately want a stronger
# model just for that harder image-reading pass.
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", OPENAI_MODEL_NAME)
OPENAI_RPM_LIMIT = int(os.getenv("OPENAI_RPM_LIMIT", "500"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Only require the API key for whichever provider is ACTUALLY
# active. This is the whole point of the switch. ---
if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
    raise ValueError("LLM_PROVIDER=gemini but GEMINI_API_KEY is not set in .env")
if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
    raise ValueError("LLM_PROVIDER=openai but OPENAI_API_KEY is not set in .env")

# --- Generic names the REST of the pipeline (llm_extractor.py, run.py)
# actually uses. Adding a third provider later means adding one more
# branch here — nowhere else in the codebase needs to change. ---
if LLM_PROVIDER == "gemini":
    PRIMARY_MODEL = GEMINI_MODEL_NAME
    FALLBACK_MODEL = GEMINI_FALLBACK_MODEL
    VISION_MODEL = GEMMA_VISION_MODEL
else:  # openai
    PRIMARY_MODEL = OPENAI_MODEL_NAME
    FALLBACK_MODEL = OPENAI_FALLBACK_MODEL
    VISION_MODEL = OPENAI_VISION_MODEL

logger.info(f"LLM provider: {LLM_PROVIDER} | primary={PRIMARY_MODEL} | "
            f"fallback={FALLBACK_MODEL} | vision={VISION_MODEL}")


# ─────────────────────────────────────────────────────────
# STEP 6: LLM CLIENTS — one singleton per provider, created lazily so we
# never import or initialize an SDK you're not actually using (e.g. a
# Gemini-only run never touches the openai package at all).
# ─────────────────────────────────────────────────────────

_gemini_client = None
_openai_client = None
_client_lock = threading.Lock()  # shared lock, protects whichever client is being created


def get_llm_client():
    """Returns the client for whichever provider is active (created once,
    reused for every call after that)."""
    global _gemini_client, _openai_client

    if LLM_PROVIDER == "gemini":
        if _gemini_client is None:
            with _client_lock:
                if _gemini_client is None:
                    from google import genai
                    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                    logger.info(f"Gemini client ready (model: {GEMINI_MODEL_NAME}, "
                                f"vision: {GEMMA_VISION_MODEL})")
        return _gemini_client

    else:  # openai
        if _openai_client is None:
            with _client_lock:
                if _openai_client is None:
                    from openai import OpenAI
                    _openai_client = OpenAI(api_key=OPENAI_API_KEY)
                    logger.info(f"OpenAI client ready (model: {OPENAI_MODEL_NAME}, "
                                f"vision: {OPENAI_VISION_MODEL})")
        return _openai_client


# ─────────────────────────────────────────────────────────
# STEP 7: RATE LIMITER — same TokenBucketLimiter class for both
# providers, just different instances/limits depending on which is active.
# ─────────────────────────────────────────────────────────

class TokenBucketLimiter:
    """Keeps requests under a per-minute limit, safely shared across workers."""

    def __init__(self, rpm_limit: int, name: str = "limiter"):
        self.rpm_limit = max(1, rpm_limit)
        self.name = name
        self._recent_calls = deque()
        self._lock = threading.Lock()

    def acquire(self):
        """Call this right before making an API request. May pause for a
        moment if we're going too fast."""
        while True:
            with self._lock:
                now = time.monotonic()
                one_minute_ago = now - 60.0
                while self._recent_calls and self._recent_calls[0] < one_minute_ago:
                    self._recent_calls.popleft()

                if len(self._recent_calls) < self.rpm_limit:
                    self._recent_calls.append(now)
                    return

                wait_time = max(0.05, 60.0 - (now - self._recent_calls[0]))

            logger.debug(f"{self.name}: slowing down for {wait_time:.2f}s to respect rate limit")
            time.sleep(wait_time)


# Gemini has separate limits for the main model vs. the Gemma vision model.
gemini_limiter = TokenBucketLimiter(GEMINI_RPM_LIMIT, name="gemini")
gemma_limiter = TokenBucketLimiter(GEMMA_RPM_LIMIT, name="gemma")

# OpenAI: one account-level limit covers both the primary and vision
# calls (same account/tier either way), so a single shared limiter.
openai_limiter = TokenBucketLimiter(OPENAI_RPM_LIMIT, name="openai")


def get_limiter(model: str) -> TokenBucketLimiter:
    """
    Picks the right rate limiter for a given model name + active
    provider. llm_extractor.py calls this instead of needing to know
    provider details itself.
    """
    if LLM_PROVIDER == "gemini":
        return gemma_limiter if "gemma" in model.lower() else gemini_limiter
    return openai_limiter


# ─────────────────────────────────────────────────────────
# STEP 8: COST CALCULATION — pricing tables for BOTH providers. Only the
# active provider's models actually get used, but keeping both here
# means switching providers never requires touching this table.
# IMPORTANT: verify these against the provider's current pricing page
# before trusting totals on a real production run — prices change.
# ─────────────────────────────────────────────────────────

PRICING_PER_MILLION_TOKENS = {
    # Gemini
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemma-4-26b-a4b-it": {"input": 0.0, "output": 0.0},   # confirm with your plan
    "gemma-4-31b-it": {"input": 0.0, "output": 0.0},       # confirm with your plan
    # OpenAI — verify against platform.openai.com/pricing before trusting
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}


def calculate_cost(input_tokens: int, output_tokens: int, model: str = None) -> float:
    """Returns the cost in dollars for one API call, whichever provider it was."""
    model = model or PRIMARY_MODEL
    prices = PRICING_PER_MILLION_TOKENS.get(model)
    if prices is None:
        logger.warning(f"No price listed for model '{model}' — showing cost as 0.0. "
                        f"Add it to PRICING_PER_MILLION_TOKENS above.")
        return 0.0
    cost = (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
    return round(cost, 6)


# ─────────────────────────────────────────────────────────
# STEP 9: STARTUP MODEL CHECK — pings only the ACTIVE provider's models.
# Better to find out a model name is wrong now than after 10,000 files.
# ─────────────────────────────────────────────────────────

def validate_models() -> None:
    """Sends one tiny test message to each model the active provider will
    use. Stops the program with a clear error if any model name is wrong
    or unavailable."""
    client = get_llm_client()
    models_to_check = {PRIMARY_MODEL, FALLBACK_MODEL, VISION_MODEL}

    for model in models_to_check:
        try:
            if LLM_PROVIDER == "gemini":
                client.models.generate_content(
                    model=model,
                    contents="ping",
                    config={"max_output_tokens": 5},
                )
            else:  # openai
                client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=5,
                )
            logger.info(f"Model check OK: {model}")
        except Exception as e:
            raise RuntimeError(
                f"Could not use model '{model}' (provider: {LLM_PROVIDER}). It may not be "
                f"available on your API key/plan — check your account's model access. "
                f"Original error: {e}"
            )