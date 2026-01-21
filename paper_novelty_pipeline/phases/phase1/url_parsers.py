"""
Phase 1 URL parsers.

Provides small, deterministic helpers to extract IDs and date hints from
academic paper URLs. These helpers keep URL parsing logic centralized and
reusable across PDF fetching and metadata enrichment.

TODO: Add more venue-specific patterns as needed.
"""

import re
from typing import Optional, Tuple


def infer_date_from_url(url: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Infer (year, month) from academic paper URL.

    Currently supports:
    - arXiv: arxiv.org/abs/2312.12345 -> (2023, 12)

    Future support planned:
    - ACL Anthology: aclweb.org/anthology/2024.acl-long.123 -> (2024, None)

    Args:
        url: Paper URL to parse

    Returns:
        Tuple of (year, month), or (None, None) if no date found.
        Month may be None if only year is available.

    Example:
        >>> infer_date_from_url("https://arxiv.org/abs/2312.12345")
        (2023, 12)
    """
    if not url:
        return None, None

    # arXiv URL pattern: /abs/YYMM.NNNNN or /pdf/YYMM.NNNNN
    if "arxiv.org" in url:
        match = re.search(r"/(abs|pdf)/(\d{2})(\d{2})\.", url)
        if match:
            year_short = int(match.group(2))
            month = int(match.group(3))
            year = 2000 + year_short
            month = month if 1 <= month <= 12 else None
            return year, month

    # TODO: Add more URL patterns when needed
    # - ACL Anthology: Extract year from ID like "2024.acl-long.123"
    # - OpenReview: Extract year from conference name in URL

    return None, None


def infer_year_from_url(url: Optional[str]) -> Optional[int]:
    """
    Infer publication year from URL (convenience wrapper).

    Args:
        url: Paper URL to parse

    Returns:
        Year as integer, or None if not found

    Example:
        >>> infer_year_from_url("https://arxiv.org/abs/2312.12345")
        2023
    """
    year, _ = infer_date_from_url(url)
    return year


def extract_openreview_forum_id(url: Optional[str]) -> Optional[str]:
    """
    Extract OpenReview forum ID from URL.

    Args:
        url: OpenReview URL (e.g., "https://openreview.net/forum?id=ABC123")

    Returns:
        Forum ID string, or None if not found

    Example:
        >>> extract_openreview_forum_id("https://openreview.net/forum?id=ABC123")
        'ABC123'
    """
    if not url or "openreview.net" not in url:
        return None

    # Match ?id=XXX or &id=XXX in URL
    match = re.search(r"[?&]id=([^&]+)", url)
    return match.group(1) if match else None


def extract_arxiv_id(url: Optional[str]) -> Optional[str]:
    """
    Extract arXiv ID from URL.

    Args:
        url: arXiv URL (e.g., "https://arxiv.org/abs/2312.12345")

    Returns:
        arXiv ID string, or None if not found

    Example:
        >>> extract_arxiv_id("https://arxiv.org/abs/2312.12345")
        '2312.12345'
    """
    if not url or "arxiv.org" not in url:
        return None

    # Match /abs/ID or /pdf/ID pattern
    match = re.search(r"arxiv\.org/(abs|pdf)/([^/\s]+)", url)
    return match.group(2) if match else None


def extract_acl_id(url: Optional[str]) -> Optional[str]:
    """
    Extract ACL Anthology ID from URL (placeholder for future).

    Args:
        url: ACL Anthology URL (e.g., "https://aclweb.org/anthology/2024.acl-long.123")

    Returns:
        Anthology ID string, or None if not found

    Example:
        >>> extract_acl_id("https://aclweb.org/anthology/2024.acl-long.123")
        '2024.acl-long.123'
    """
    if not url or "aclweb.org" not in url:
        return None

    # Match anthology/ID pattern
    match = re.search(r"anthology/([^/\s]+)", url)
    return match.group(1) if match else None
