"""
config.py – Central configuration for Gemini + Gemma vision fallback pipeline.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────
ITEM_CODE_MIN = 1000000
ITEM_CODE_MAX = 4999999

# ─── Environment ──────────────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16000"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0.3"))
INPUT_DIR = os.getenv("INPUT_DIR", "./input")

# NEW — Gemma vision fallback model
GEMMA_VISION_MODEL = os.getenv("GEMMA_VISION_MODEL", "gemma-4-26b-a4b-it")

# ─── API Key ──────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set in .env")

# ─── Gemini Client ─────────────────────────────────────────────────
_gemini_client = None

def get_llm_client():
    """Return the Gemini client (singleton). Also used for Gemma vision calls
    since both are served through the same Google AI Studio API."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"Gemini client ready (model: {MODEL_NAME}, gemma vision: {GEMMA_VISION_MODEL})")
    return _gemini_client

def calculate_cost(input_tokens, output_tokens):
    # Gemini free tier – cost is zero
    return 0.0