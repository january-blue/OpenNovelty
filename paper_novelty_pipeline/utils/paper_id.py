"""
Utilities for constructing a globally canonical paper id.

Strategy: Use MD5 hash of normalized title (32 characters).
All papers get a uniform canonical_id format based solely on their title.
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional


def _normalize_title_for_hash(title: Optional[str]) -> str:
    """
    Normalize title for MD5 hashing.
    
    Steps:
    1. Convert to lowercase
    2. Remove all non-alphanumeric characters (spaces, punctuation, etc.)
    
    Example:
        "Attention Is All You Need" -> "attentionisallyouneed"
    """
    if not title:
        return ""
    t = title.lower()
    # Keep only alphanumeric characters (letters and digits)
    t = re.sub(r'[^a-z0-9]', '', t)
    return t


def make_canonical_id(
    *,
    title: Optional[str] = None,
    paper_id: Optional[str] = None,
    url: Optional[str] = None,
    **kwargs  # Ignore all other parameters (doi, arxiv_id, year, etc.)
) -> str:
    """
    Make a canonical, globally consistent paper id based on title MD5 hash.
    
    All external identifiers (DOI, ArXiv, OpenReview) are ignored.
    Only the paper title is used for canonical_id generation.
    
    Returns:
        32-character MD5 hash of normalized title, or fallback hash if title is missing.
    
    Examples:
        >>> make_canonical_id(title="Attention Is All You Need")
        '592b2b0bae57a89f301fcf342baf118a'  # Example hash
        
        >>> make_canonical_id(title="Deep Learning: A Survey")
        '13d60d58cf8c1891e2f6e8a9b5c4d3e2'  # Example hash
    """
    # 1) Title MD5 (primary strategy)
    if title:
        norm = _normalize_title_for_hash(title)
        if norm:
            return hashlib.md5(norm.encode("utf-8")).hexdigest()  # Full 32-char MD5
    
    # 2) Fallback: hash of whatever we have (paper_id or url)
    fallback = paper_id or url or "unknown"
    return hashlib.md5(str(fallback).encode("utf-8")).hexdigest()
