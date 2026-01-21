"""
Phase 3: Utility Functions

Pure utility functions with no class dependencies.
"""

import json
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from paper_novelty_pipeline.models import RetrievedPaper


# ============================================================================
# Text Sanitization
# ============================================================================

def sanitize_filename(text: str, max_len: int = 25) -> str:
    """
    Convert text to a filesystem-safe filename component.
    
    - Converts to lowercase
    - Normalizes separators (hyphens, slashes) to underscores
    - Removes special characters
    - Truncates at word boundaries
    
    Args:
        text: Original text
        max_len: Maximum length (default: 25)
        
    Returns:
        Safe filename component
    """
    if not text:
        return ""
    
    # Normalize: lowercase, separators to spaces, remove special chars
    text = text.lower()
    text = re.sub(r'[-_/]+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]+', '', text)
    text = re.sub(r'\s+', '_', text.strip())
    
    # Truncate at word boundary if needed
    if len(text) > max_len:
        text = text[:max_len]
        last_underscore = text.rfind('_')
        if last_underscore > max_len // 2:
            text = text[:last_underscore]
    
    return text


def sanitize_id(text: str, max_len: int = 80) -> str:
    """
    Create a filesystem-safe identifier from text.
    
    Simpler than sanitize_filename - just replaces invalid chars with underscores.
    Used for paper IDs and longer identifiers.
    """
    if not text:
        return "unknown"
    
    try:
        s = re.sub(r"[^\w\-.]+", "_", text)
        return s[:max_len] if len(s) > max_len else s
    except Exception:
        return "unknown"


# ============================================================================
# JSON Parsing
# ============================================================================

def parse_json_flexible(text: Any) -> Optional[Union[dict, list]]:
    """
    Parse JSON from various formats (string, dict, list, code blocks).
    
    Handles:
    - Direct dict or list
    - JSON string (dict or list)
    - Code blocks with ```json
    - Partial JSON (first { to last } or first [ to last ])
    """
    # Handle direct dict or list
    if isinstance(text, (dict, list)):
        return text
    
    if not isinstance(text, str):
        return None
    
    # Try direct parse (handles both dict and list)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        pass
    
    # Try code block extraction for dict
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    # Try code block extraction for list
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    
    # Try partial extraction for dict (first { to last })
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    
    # Try partial extraction for list (first [ to last ])
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    
    return None


def read_json_list(path: Path) -> List[Dict[str, Any]]:
    """
    Read JSON file that may contain a list or dict with list values.
    
    Supports formats:
    - Direct list: [{"id": 1}, ...]
    - Dict with list: {"items": [...], "candidates": [...], ...}
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    
    if isinstance(obj, dict):
        # Try common list keys
        for key in ("items", "candidates", "merged_dedup", "merged", "data", "results"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value
    
    return []


# ============================================================================
# Paper ID Utilities
# ============================================================================

def get_short_paper_id(paper_id: str) -> str:
    """
    Generate a short, filesystem-safe ID from a paper ID.
    
    - If already short and safe, return as-is
    - If contains arXiv ID, extract it
    - Otherwise, use MD5 hash (first 12 chars)
    """
    # Already short and safe
    if len(paper_id) <= 15 and re.match(r'^[A-Za-z0-9_-]+$', paper_id):
        return paper_id
    
    # Extract arXiv ID
    match = re.search(r'(\d{4}\.\d{4,5})', paper_id)
    if match:
        return f"arxiv_{match.group(1).replace('.', '_')}"
    
    # Use hash
    return hashlib.md5(paper_id.encode()).hexdigest()[:12]


def normalize_title(title: Optional[str]) -> Optional[str]:
    """Normalize title for comparison (lowercase, stripped)."""
    return title.lower().strip() if title else None


# ============================================================================
# Paper Object Construction
# ============================================================================

def extract_authors(item: Dict[str, Any]) -> List[str]:
    """
    Extract author names from Phase2 JSON item.
    
    Handles multiple formats with priority:
    1. authorships[].author_name (most accurate)
    2. raw_metadata.authors (string, comma-separated)
    3. item.authors (array, may be incorrectly split)
    
    Returns deduplicated list preserving order.
    """
    authors = []
    raw_meta = item.get("raw_metadata") or {}
    
    # Priority 1: authorships (most accurate)
    if isinstance(raw_meta, dict) and "authorships" in raw_meta:
        authorships = raw_meta.get("authorships") or []
        if isinstance(authorships, list):
            for a in authorships:
                if isinstance(a, dict):
                    name = a.get("author_name")
                    if name and isinstance(name, str):
                        name = name.strip()
                        if name:
                            authors.append(name)
    
    # Priority 2: raw_metadata.authors (string)
    if not authors and isinstance(raw_meta, dict):
        authors_str = raw_meta.get("authors")
        if authors_str:
            parts = re.split(r"[,;]|\band\b", str(authors_str))
            authors = [p.strip() for p in parts if p.strip()]
    
    # Priority 3: item.authors (array, may be incorrectly split)
    if not authors:
        authors_raw = item.get("authors") or item.get("author_names") or item.get("metadata", {}).get("authors") or []
        if isinstance(authors_raw, list):
            authors = [str(a).strip() for a in authors_raw if str(a).strip()]
        elif isinstance(authors_raw, str):
            parts = re.split(r"[,;]|\band\b", authors_raw)
            authors = [p.strip() for p in parts if p.strip()]
    
    # Deduplicate (case-insensitive, preserve order)
    seen = set()
    unique = []
    for author in authors:
        key = author.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(author.strip())
    
    return unique


def make_retrieved_paper(item: Dict[str, Any], *, fallback_score: float = 0.0) -> Optional[RetrievedPaper]:
    """
    Convert Phase2 JSON item to RetrievedPaper object.
    
    Handles various metadata formats and field locations.
    """
    try:
        meta = item.get("metadata") or {}
        
        # Extract fields with fallbacks
        title = item.get("title") or meta.get("title") or ""
        abstract = item.get("abstract") or meta.get("abstract") or meta.get("paper_abstract") or ""
        authors = extract_authors(item)  # Use dedicated function
        venue = item.get("venue") or meta.get("venue") or ""
        year = item.get("year") or meta.get("year") or 0
        doi = item.get("doi") or meta.get("doi")
        url = item.get("url_pref") or item.get("url") or meta.get("url")
        pdf_url = item.get("pdf_url") or meta.get("pdf_url")
        
        # Paper ID with fallbacks
        paper_id = item.get("paper_id") or meta.get("paper_id") or doi or url or title
        score = item.get("relevance_score") or item.get("final_score") or fallback_score or 0.0
        
        # Canonical ID (prioritize canonical_id field, fallback to generation if needed)
        canonical_id = item.get("canonical_id") or meta.get("canonical_id")
        if not canonical_id:
            # Fallback: generate canonical_id if not present (for backward compatibility)
            from paper_novelty_pipeline.utils.paper_id import make_canonical_id
            canonical_id = make_canonical_id(
                paper_id=paper_id,
                doi=doi,
                arxiv_id=item.get("arxiv_id"),
                url=url,
                pdf_url=pdf_url,
                title=title,
                year=year,
            )
        
        # Type conversions
        year_int = int(year) if (isinstance(year, int) or (isinstance(year, str) and year.isdigit())) else 0
        raw_meta = item.get("raw_metadata") or {}
        
        return RetrievedPaper(
            paper_id=str(paper_id),
            title=str(title or ""),
            abstract=str(abstract or ""),
            authors=authors,
            venue=str(venue or ""),
            year=year_int,
            doi=str(doi) if doi else None,
            arxiv_id=None,
            relevance_score=float(score),
            pdf_url=str(pdf_url) if pdf_url else None,
            source_url=str(url) if url else None,
            raw_metadata=raw_meta if isinstance(raw_meta, dict) and raw_meta else None,
            canonical_id=canonical_id,
        )
    except Exception:
        return None


# ============================================================================
# Filename Generation
# ============================================================================

def get_comparison_filename(
    index: int,
    aspect: str,
    contribution_name: str,
    candidate_id: str
) -> str:
    """
    Generate descriptive filename for comparison result.
    
    Format: {index:02d}_{aspect}_{contribution}_{candidate}.json
    Example: 01_contribution_residual_weight_adapt_arxiv_2509_08755.json
    """
    aspect_safe = sanitize_filename(aspect, max_len=20)
    contribution_safe = sanitize_filename(contribution_name, max_len=25)
    candidate_short = get_short_paper_id(candidate_id)[:15]
    
    return f"{index:02d}_{aspect_safe}_{contribution_safe}_{candidate_short}.json"

