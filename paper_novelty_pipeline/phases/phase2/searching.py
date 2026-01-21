"""
Phase 2: Paper Searching.

This module handles the academic paper search phase of the novelty analysis pipeline.

Responsibilities:
  1. Prepare search queries from Phase 1 extracted content (core_task + contributions)
  2. Execute concurrent searches via Wispaper API
  3. Save raw API responses to phase2/raw_responses/
  4. Return execution statistics

Note:
  This module depends on Wispaper API which will be publicly available in a future release.
"""

from pathlib import Path
from typing import Dict, Any, Optional

from paper_novelty_pipeline.models import ExtractedContent


class PaperSearcher:
    """Phase 2: Search for related papers via Wispaper API.
    
    Note: Wispaper API will be publicly available in a future release.
    """
    
    def __init__(self, concurrency: Optional[int] = None):
        raise NotImplementedError(
            "Phase 2 searching depends on Wispaper API which is not yet publicly available. "
            "Please check the project repository for updates on API access."
        )
    
    def search_all(
        self,
        extracted: ExtractedContent,
        out_dir: Path,
    ) -> Dict[str, Any]:
        """Execute all searches for a paper."""
        raise NotImplementedError("Wispaper API is not yet publicly available.")


def run_phase2_search(
    extracted: ExtractedContent,
    out_dir: Path,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Run Phase2 search (API calls only).
    
    Note: Wispaper API will be publicly available in a future release.
    """
    raise NotImplementedError(
        "Phase 2 searching depends on Wispaper API which is not yet publicly available. "
        "Please check the project repository for updates on API access."
    )
