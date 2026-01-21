"""
Centralized path utilities for paper novelty pipeline.

Provides:
- safe_dir_name: Generate safe directory names from paper URLs/IDs
- Standard output file path constants (single source of truth)
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ============================================================================
# File name constants (single source of truth)
# ============================================================================

# Phase 1 outputs
PAPER_JSON = "paper.json"
PHASE1_EXTRACTED_JSON = "phase1_extracted.json"
PUB_DATE_JSON = "pub_date.json"

# Phase 2 outputs
PHASE2_SEARCH_RESULTS_JSON = "phase2_search_results.json"


# ============================================================================
# safe_dir_name: Generate safe directory names from paper URLs/IDs
# ============================================================================

def safe_dir_name(paper_id: str) -> str:
    """
    Generate a safe directory name from paper_id (OpenReview/arXiv/ACL/IJCAI/local path).

    This is the single canonical implementation. All entrypoints should use this.
    """
    if not paper_id:
        return "paper"

    # 1. arXiv URL pattern
    if "arxiv.org" in paper_id:
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([\d.]+)", paper_id)
        if m:
            return f"arxiv_{m.group(1).replace('.', '_')}"

    # 2. OpenReview URL pattern
    if "openreview.net" in paper_id:
        m = re.search(r"[?&]id=([^&]+)", paper_id)
        if m:
            return f"openreview_{m.group(1)}"

    # 3. ACL Anthology pattern (e.g., https://aclanthology.org/2024.acl-long.575.pdf)
    if "aclanthology.org" in paper_id:
        # Extract the identifier (e.g., 2024.acl-long.575)
        # Usually it's the part before .pdf or the last segment
        clean_id = paper_id.split('/')[-1].replace('.pdf', '')
        if clean_id:
            return f"acl_{clean_id.replace('.', '_')}"

    # 4. IJCAI pattern (e.g., https://www.ijcai.org/proceedings/2025/1108.pdf)
    if "ijcai.org" in paper_id:
        # Usually it's the last segment (e.g., 1108)
        clean_id = paper_id.split('/')[-1].replace('.pdf', '')
        if clean_id:
            return f"ijcai_{clean_id}"

    # 5. Local file path (check if it exists or looks like one)
    # Use Path(paper_id).exists() if possible, but be careful with URLs
    if not paper_id.startswith("http"):
        path_obj = Path(paper_id)
        if path_obj.suffix.lower() == ".pdf" or path_obj.exists():
            return f"local_{path_obj.stem}"

    # Fallback: sanitize and truncate
    # If it's a URL, use the last part
    if paper_id.startswith("http"):
        last_part = paper_id.split('/')[-1].split('?')[0]
        paper_id = last_part or paper_id

    safe = re.sub(r"[^\w\-_\.]", "_", paper_id)[:50]
    return safe or "paper"


# Note: Additional path helper classes and unused constants were removed because
# they were not referenced by the entrypoints or CLI surface.
