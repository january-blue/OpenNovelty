"""
Helpers for computing deterministic artifact filenames across phases.

Goal:
- Phase3 can pre-compute Phase4 output filenames and write them into JSON.
- Phase4 uses the same logic so filenames match even if Phase4 runs later.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional


def safe_filename(text: str, *, max_len: int = 50) -> str:
    """Generate a safe filename from text."""
    if not text:
        return "Unknown"
    # Remove special characters, keep only alphanumeric, underscore, dash, and spaces.
    safe = re.sub(r"[^\w\s-]", "", str(text))
    # Replace spaces/dashes with underscores
    safe = re.sub(r"[-\s]+", "_", safe).strip("_")
    if not safe:
        safe = "Unknown"
    return safe[:max_len]


def yyyymmdd_from_iso(iso_ts: Optional[str]) -> Optional[str]:
    """Parse an ISO timestamp and return YYYYMMDD, or None if parsing fails."""
    if not iso_ts or not isinstance(iso_ts, str):
        return None
    s = iso_ts.strip()
    if not s:
        return None
    # Support common "Z" suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
        return dt.strftime("%Y%m%d")
    except Exception:
        return None


def phase4_lightweight_report_basename_from_phase3_report(data: Dict[str, Any]) -> str:
    """
    Deterministically compute the Phase4 lightweight report basename.

    Format:
      novelty_report_lightweight_{safe_title[:50]}_{date_str}

    date_str preference order:
    - Phase3 metadata.generated_at (stable per Phase3 report)
    - fallback to current date (keeps old behavior if metadata missing)
    """
    original = (data or {}).get("original_paper", {}) or {}
    title = original.get("title") or "Unknown"
    safe_title = safe_filename(title, max_len=50)[:50]

    meta = (data or {}).get("metadata", {}) or {}
    date_str = yyyymmdd_from_iso(meta.get("generated_at")) or datetime.now().strftime("%Y%m%d")

    return f"novelty_report_lightweight_{safe_title}_{date_str}"


def phase4_lightweight_pdf_filename_from_phase3_report(data: Dict[str, Any]) -> str:
    return phase4_lightweight_report_basename_from_phase3_report(data) + ".pdf"


def phase4_lightweight_md_filename_from_phase3_report(data: Dict[str, Any]) -> str:
    return phase4_lightweight_report_basename_from_phase3_report(data) + ".md"





