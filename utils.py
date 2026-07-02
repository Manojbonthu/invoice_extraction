"""
utils.py – Miscellaneous helper functions for the invoice extraction project.
Provides reusable utilities for file handling, retry logic, text cleaning, and date parsing.
"""

import os
import re
import time
import functools
from typing import Any, Callable, Optional, List, Dict

from config import logger


# =====================================================================
# FILE SYSTEM HELPERS
# =====================================================================

def ensure_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        path: Directory path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.debug(f"Created directory: {path}")


def sanitize_filename(filename: str) -> str:
    """
    Remove invalid characters from a filename for safe file system usage.
    
    Args:
        filename: Original filename (e.g., "invoice:123.pdf").
    
    Returns:
        Safe filename (e.g., "invoice_123.pdf").
    """
    # Replace Windows/Linux invalid characters with underscore
    return re.sub(r'[\\/*?:"<>|]', '_', filename).strip()


def get_file_extension(filepath: str) -> str:
    """Return the file extension (lowercase) without the dot."""
    ext = os.path.splitext(filepath)[1]
    return ext.lower().lstrip('.')


# =====================================================================
# RETRY DECORATOR (for API calls, network operations)
# =====================================================================

def retry_on_exception(
    max_retries: int = 3,
    delay: int = 2,
    backoff_factor: int = 2,
    exceptions: tuple = (Exception,),
):
    """
    Decorator to retry a function on exception with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts.
        delay: Initial delay in seconds.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch and retry.
    
    Example:
        @retry_on_exception(max_retries=5, delay=1)
        def call_openai():
            # ... API call ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(
                            f"Retry {attempt+1}/{max_retries} for {func.__name__} "
                            f"in {wait_time:.1f}s due to: {e}"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"All {max_retries} retries failed for {func.__name__}: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator


# =====================================================================
# TEXT CLEANING (used in OCR grid reconstruction)
# =====================================================================

def clean_text(text: str) -> str:
    """
    Remove control characters and collapse multiple spaces.
    
    Args:
        text: Raw text from OCR.
    
    Returns:
        Cleaned text (single spaces, no control characters).
    """
    if not text:
        return ""
    # Remove non-printable characters (except newline and tab)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Collapse multiple spaces/tabs/newlines into a single space
    return " ".join(text.split())


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace (spaces, tabs, newlines) to a single space.
    """
    if not text:
        return ""
    return " ".join(text.split())


def extract_digits(text: str) -> str:
    """
    Extract only digits from a string. Useful for HSN/Item Code cleaning.
    
    Example: "8482 99 00" -> "84829900"
    """
    if not text:
        return ""
    return re.sub(r'\D', '', text)  # \D = non-digit


# =====================================================================
# DATE NORMALIZATION
# =====================================================================

def normalize_date(date_str: str) -> str:
    """
    Attempt to normalize various date formats to dd/mm/yyyy.
    
    Handles:
        - dd.mm.yyyy  -> dd/mm/yyyy
        - dd/mm/yyyy  -> dd/mm/yyyy
        - yyyy-mm-dd  -> dd/mm/yyyy
        - dd-mm-yyyy  -> dd/mm/yyyy
    
    Args:
        date_str: Raw date string from OCR.
    
    Returns:
        Normalized date in dd/mm/yyyy format, or empty string if invalid.
    """
    if not date_str:
        return ""
    
    # Clean and extract numbers
    date_str = normalize_whitespace(date_str)
    # Replace dots, hyphens with slashes
    date_str = re.sub(r'[.-]', '/', date_str)
    
    parts = date_str.split('/')
    if len(parts) != 3:
        return date_str  # Return as-is if not a clear date
    
    # Check if format is yyyy/mm/dd
    if len(parts[0]) == 4 and parts[0].isdigit():
        # Convert yyyy/mm/dd to dd/mm/yyyy
        return f"{parts[2]}/{parts[1]}/{parts[0]}"
    
    # If day/month are swapped? We assume dd/mm/yyyy by default.
    # Validate that day, month, year are all digits.
    if all(p.isdigit() for p in parts):
        return date_str
    return date_str


# =====================================================================
# DICTIONARY HELPERS
# =====================================================================

def safe_get(data: Dict, keys: List[str], default: Any = None) -> Any:
    """
    Safely access nested dictionary keys.
    
    Args:
        data: Dictionary to traverse.
        keys: List of keys in order (e.g., ['Invoice Items', 0, 'hsn_sac']).
        default: Value to return if key path is missing.
    
    Example:
        hsn = safe_get(invoice, ['Invoice Items', 0, 'hsn_sac'], default=None)
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return default
    return current


# =====================================================================
# BATCH HELPERS
# =====================================================================

def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Split a list into smaller chunks of a given size.
    
    Args:
        lst: List to split.
        chunk_size: Max size of each chunk.
    
    Returns:
        List of chunks.
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def count_total_pages(pdf_paths: List[str]) -> int:
    """
    Calculate total pages across multiple PDFs.
    Uses fitz (PyMuPDF) – requires import inside to avoid dependency if not used.
    """
    try:
        import fitz
        total = 0
        for path in pdf_paths:
            try:
                doc = fitz.open(path)
                total += doc.page_count
                doc.close()
            except Exception:
                continue
        return total
    except ImportError:
        logger.warning("fitz (PyMuPDF) not available for page counting.")
        return 0


# =====================================================================
# ENVIRONMENT HELPERS
# =====================================================================

def is_running_in_colab() -> bool:
    """Check if the code is running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def get_memory_usage() -> str:
    """
    Get current memory usage (platform independent).
    Works on Windows, Linux, macOS.
    """
    try:
        import psutil
        process = psutil.Process()
        mem = process.memory_info()
        return f"{mem.rss / 1024**3:.2f} GB"
    except ImportError:
        return "N/A (install psutil)"
    except Exception:
        return "Unknown"


# =====================================================================
# DEBUG HELPERS
# =====================================================================

def print_json_pretty(data: Dict, max_depth: int = 3) -> None:
    """
    Pretty-print JSON data up to a certain depth (to avoid terminal flooding).
    """
    import json
    
    def truncate(obj, depth):
        if depth > max_depth:
            return "... (truncated)"
        if isinstance(obj, dict):
            return {k: truncate(v, depth + 1) for k, v in obj.items()}
        if isinstance(obj, list):
            if len(obj) > 5:
                return [truncate(obj[0], depth + 1), f"... ({len(obj)-2} more) ...", truncate(obj[-1], depth + 1)]
            return [truncate(item, depth + 1) for item in obj]
        return obj
    
    truncated = truncate(data, 0)
    print(json.dumps(truncated, indent=2, default=str))