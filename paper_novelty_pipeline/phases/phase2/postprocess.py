#!/usr/bin/env python3
"""
Phase2 post-processing module.

Reads raw_responses/ and generates:
1. candidates/ intermediate files (with queries and stats)
2. final/ TopK files (deduplicated using canonical_id)
3. final/citation_index.json (global dedup, LLM aliases, roles)
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from paper_novelty_pipeline.utils.paper_id import make_canonical_id
from paper_novelty_pipeline.utils.paths import PAPER_JSON

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

# Default TopK values for candidate selection
PHASE2_TOPK_CORE_TASK = 50
PHASE2_TOPK_CONTRIBUTION = 10

# Scope identifiers
SCOPE_CORE_TASK = "core_task"
SCOPE_CONTRIBUTION_PREFIX = "contribution_"

# Role type identifiers (used in citation_index.json)
ROLE_ORIGINAL_PAPER = "original_paper"
ROLE_CORE_TASK = "core_task"
ROLE_CONTRIBUTION = "contribution"

# Canonical ID prefixes and their quality priorities (higher = better)
# Real DOI (excluding arXiv DOIs which use 10.48550) has highest priority
CANONICAL_ID_PRIORITY = {
    "doi:": 4,       # Real DOI (not arXiv DOI)
    "arxiv:": 3,     # ArXiv ID
    "openreview:": 2,  # OpenReview ID
    "title:": 1,     # Title-based hash (fallback)
}
ARXIV_DOI_PREFIX = "10.48550"  # arXiv DOIs start with this


# ===========================================================================
# Helper Functions
# ===========================================================================

def _read_json(path: Path) -> Any:
    """Read JSON file."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: Any) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _is_perfect(item: Dict[str, Any]) -> bool:
    """Check if paper is marked as 'perfect'."""
    if item.get("perfect") is True:
        return True
    flags = item.get("flags")
    if isinstance(flags, dict) and flags.get("perfect") is True:
        return True
    return False


def _score(item: Dict[str, Any]) -> float:
    """Extract relevance score from paper."""
    for key in ("relevance_score", "final_score", "score"):
        score_value = item.get(key)
        if score_value is None:
            continue
        try:
            return float(score_value)
        except (TypeError, ValueError):
            continue
    return 0.0


_ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)
_ARXIV_STD_RE = re.compile(r"^(\d{4})\.(\d{4,5})$", re.IGNORECASE)


def _norm_doi(doi: Optional[str]) -> Optional[str]:
    """Normalize DOI string."""
    if not doi:
        return None
    doi_str = doi.strip().lower()
    doi_str = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi_str)
    doi_str = re.sub(r"^doi:", "", doi_str)
    return doi_str


def _extract_arxiv_id_from_canonical(canonical_id: str) -> Optional[str]:
    """Extract arxiv_id from canonical_id if it's in arxiv:* format."""
    if not canonical_id or not canonical_id.startswith("arxiv:"):
        return None
    return canonical_id.split(":", 1)[1] if ":" in canonical_id else None


def _norm_title(title: str) -> str:
    """
    Normalize title for deduplication and comparison.
    
    Transforms title to lowercase, keeps only alphanumeric characters and spaces,
    and collapses multiple whitespace into single spaces.
    
    Note:
        This function uses [^a-z0-9 ]+ pattern. The _generate_citation_index method
        uses a slightly different inline pattern [^\\w\\s] which also preserves
        underscores. Both achieve the same goal of title-based deduplication but
        are kept separate to maintain backward compatibility.
    
    Args:
        title: Raw paper title
        
    Returns:
        Normalized title string for comparison
        
    Example:
        >>> _norm_title("  BERT: Pre-training of Deep Bidirectional Transformers  ")
        'bert pre training of deep bidirectional transformers'
    """
    if not title:
        return ""
    norm_title = title.strip().lower()
    norm_title = re.sub(r"[^a-z0-9 ]+", " ", norm_title)
    norm_title = re.sub(r"\s+", " ", norm_title)
    return norm_title.strip()


def _calculate_quality_flags(verdict: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """
    Calculate quality flags based on Wispaper's official assessment rules.
    
    This function interprets the criteria_assessment from Wispaper's verification
    response and produces boolean flags indicating paper quality levels.
    
    Rules:
        Single-criterion (len(criteria) == 1):
            - 'support' → perfect=True
            - 'somewhat_support' → partial=True (no type check needed)
            - Other assessments → no=True
            
        Multi-criterion (len(criteria) > 1):
            - All 'support' → perfect=True
            - Any ('support' OR 'somewhat_support') AND type != 'time' → partial=True
            - Otherwise → no=True
    
    Args:
        verdict: Parsed JSON dict from Wispaper verification response,
                 containing 'criteria_assessment' list
                 
    Returns:
        Dict with keys 'perfect', 'partial', 'no' (exactly one will be True)
        
    Example:
        >>> _calculate_quality_flags({'criteria_assessment': [{'assessment': 'support'}]})
        {'perfect': True, 'partial': False, 'no': False}
    """
    flags = {"perfect": False, "partial": False, "no": False}
    
    if not verdict or not isinstance(verdict.get('criteria_assessment'), list):
        return flags
    
    criteria = [a for a in verdict['criteria_assessment'] if isinstance(a, dict)]
    if not criteria:
        return flags
        
    assessments = [a.get('assessment') for a in criteria]
    
    # Check if all criteria are 'support' → perfect
    if assessments and all(a == 'support' for a in assessments):
        return {"perfect": True, "partial": False, "no": False}
    
    # Single-criterion case
    if len(criteria) == 1:
        if assessments[0] == 'somewhat_support':
            return {"perfect": False, "partial": True, "no": False}
        return {"perfect": False, "partial": False, "no": True}
    
    # Multi-criterion case: check for partial match (excluding 'time' type criteria)
    if any(
        a.get('assessment') in ('support', 'somewhat_support') and a.get('type') != 'time'
        for a in criteria
    ):
        return {"perfect": False, "partial": True, "no": False}
    
    return {"perfect": False, "partial": False, "no": True}


def _ensure_openreview_id(paper: Dict[str, Any]) -> Optional[str]:
    """
    Extract and normalize OpenReview ID from paper metadata.
    
    Ensures the returned ID has the 'openreview:' prefix for consistency.
    Checks multiple possible source fields in priority order.
    
    Args:
        paper: Paper metadata dictionary
        
    Returns:
        Normalized OpenReview ID with 'openreview:' prefix, or None if not found
        
    Example:
        >>> _ensure_openreview_id({'openreview_id': 'ABC123'})
        'openreview:ABC123'
        >>> _ensure_openreview_id({'openreview_forum_id': 'XYZ789'})
        'openreview:XYZ789'
    """
    # First check if openreview_id is already present
    openreview_id = paper.get("openreview_id")
    
    if not openreview_id:
        # Fallback: try to find forum_id from various sources
        forum_id = (
            paper.get("openreview_forum_id")
            or paper.get("raw_metadata", {}).get("openreview_forum_id")
            or paper.get("raw_metadata", {}).get("id")
        )
        if forum_id:
            return f"openreview:{forum_id}"
        return None
    
    # Ensure prefix is present
    if not openreview_id.startswith("openreview:"):
        return f"openreview:{openreview_id}"
    
    return openreview_id


# ============================================================================
# Phase2Processor Class
# ============================================================================

class Phase2Processor:
    """
    Complete Phase2 post-processing pipeline.
    
    Workflow:
    1. Parse raw_responses/ and group by scope
    2. For each scope:
       - Filter (perfect only)
       - Attach canonical_id
       - Deduplicate (scope-internal, by canonical_id)
       - Remove self-reference
       - Temporal filtering
       - Generate candidates/ intermediate files
       - Generate final/ TopK files
    3. Generate citation_index.json:
       - Global deduplication (title + year)
       - Merge canonical_ids
       - Generate roles
       - LLM alias generation
    """

    def __init__(
        self,
        phase2_dir: Path,
        cutoff_year: Optional[int] = None,
        self_pdf_url: Optional[str] = None,
        self_title: Optional[str] = None,
        topk_core_task: int = PHASE2_TOPK_CORE_TASK,
        topk_contribution: int = PHASE2_TOPK_CONTRIBUTION,
    ):
        self.phase2_dir = Path(phase2_dir)
        self.raw_responses_dir = self.phase2_dir / "raw_responses"
        self.candidates_dir = self.phase2_dir / "candidates"
        self.final_dir = self.phase2_dir / "final"
        
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)

        self.cutoff_year = cutoff_year
        self.self_pdf_url = self_pdf_url
        self.self_title = self_title
        self.topk_core_task = topk_core_task
        self.topk_contribution = topk_contribution
        
        # Load original paper info and generate canonical_id for self-filtering
        self.original_paper_canonical_id = self._load_original_paper_canonical_id()

        logger.info(f"Phase2Processor initialized for {phase2_dir}")
        logger.info(f"  Cutoff Year: {self.cutoff_year}")
        logger.info(f"  Self PDF URL: {self.self_pdf_url}")
        logger.info(f"  Self Title: {self.self_title}")
        logger.info(f"  Original Paper canonical_id: {self.original_paper_canonical_id}")
        logger.info(f"  TopK Core Task: {self.topk_core_task}")
        logger.info(f"  TopK Contribution: {self.topk_contribution}")

    def _load_original_paper_canonical_id(self) -> Optional[str]:
        """
        Load original paper info from Phase1 and generate its canonical_id.
        This is used for self-reference filtering.
        
        Returns:
            canonical_id of the original paper, or None if not found
        """
        # Try to find Phase1 directory
        base_dir = self.phase2_dir.parent
        phase1_dir = base_dir / "phase1"
        
        if not phase1_dir.exists():
            logger.warning("Phase1 directory not found, cannot load original paper info for self-filtering.")
            return None
        
        # Try to read paper.json
        paper_json_path = phase1_dir / "paper.json"
        if not paper_json_path.exists():
            logger.warning("Phase1 paper.json not found, cannot load original paper info for self-filtering.")
            return None
        
        try:
            with open(paper_json_path, 'r', encoding='utf-8') as f:
                paper_data = json.load(f)
            
            # Generate canonical_id from the original paper's title
            title = paper_data.get('title')
            if not title:
                logger.warning("Original paper has no title, cannot generate canonical_id for self-filtering.")
                return None
            
            # Use make_canonical_id to ensure consistency with candidate papers
            canonical_id = make_canonical_id(title=title)
            logger.info(f"Loaded original paper canonical_id for self-filtering: {canonical_id}")
            return canonical_id
            
        except Exception as e:
            logger.warning(f"Failed to load original paper info: {e}")
            return None

    def process(self) -> Dict[str, Any]:
        """
        Main entry point: complete Phase2 post-processing.
        
        Returns:
            Statistics dictionary
        """
        logger.info(f"Starting Phase2 processing for {self.phase2_dir}")

        # Step 1: Parse raw responses and group by scope
        papers_by_scope = self._parse_raw_responses()
        total_papers_before_filter = sum(len(p) for p in papers_by_scope.values())
        logger.info(f"Parsed {total_papers_before_filter} raw papers across {len(papers_by_scope)} scopes.")

        # Step 2: Process each scope independently
        all_topk_papers: Dict[str, List[Dict]] = {}
        all_candidates: Dict[str, Dict] = {}
        total_papers_after_filter = 0

        for scope, papers in papers_by_scope.items():
            topk_limit = self.topk_core_task if scope == SCOPE_CORE_TASK else self.topk_contribution
            logger.info(f"Processing scope '{scope}' with {len(papers)} papers, TopK limit: {topk_limit}")
            
            # Filter and deduplicate
            processed_papers, stats = self._process_scope(scope, papers, topk_limit)
            all_topk_papers[scope] = processed_papers
            all_candidates[scope] = stats
            total_papers_after_filter += len(processed_papers)
            
            logger.info(f"Scope '{scope}' resulted in {len(processed_papers)} TopK papers.")

        # Step 3: Generate candidates/ intermediate files
        self._write_candidates_files(all_candidates)

        # Step 4: Generate citation_index from all TopK papers (determines final canonical_ids)
        citation_index = self._generate_citation_index(all_topk_papers)
        logger.info(f"Generated citation_index with {len(citation_index.get('items', []))} unique papers.")

        # Step 5: Write citation_index (Single Source of Truth)
        self._write_citation_index(citation_index)

        # Step 6: Derive TopK files from citation_index (ensures consistency)
        derived_files = self._derive_topk_from_citation_index(citation_index)
        logger.info(f"Derived {len(derived_files)} TopK files from citation_index.")

        # Step 7: Write metadata files for Phase3 compatibility
        self._write_metadata_files(citation_index, all_candidates, derived_files)

        logger.info("Phase2 processing completed.")

        return {
            'total_scopes': len(papers_by_scope),
            'total_papers_before_filter': total_papers_before_filter,
            'total_papers_after_filter': total_papers_after_filter,
            'citation_index_size': len(citation_index.get('items', [])),
            'derived_topk_files': derived_files,
            'scopes': {
                scope: {
                    'papers_before': len(papers_by_scope[scope]),
                    'papers_after_filter': len(all_topk_papers.get(scope, [])),
                    'papers_in_topk': derived_files.get(scope, 0),
                    'topk_limit': (self.topk_core_task if scope == SCOPE_CORE_TASK else self.topk_contribution)
                }
                for scope in papers_by_scope
            }
        }

    def _parse_raw_responses(self) -> Dict[str, List[Dict]]:
        """
        Scans raw_responses directory, parses JSON, and groups papers by scope.
        Also extracts the verdict for flags generation.
        """
        papers_by_scope: Dict[str, List[Dict]] = defaultdict(list)
        
        if not self.raw_responses_dir.exists():
            logger.warning(f"raw_responses directory not found: {self.raw_responses_dir}")
            return papers_by_scope
            
        for raw_file in self.raw_responses_dir.glob("raw_*.json"):
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                
                scope = self._identify_scope(raw_file.name)
                if not scope:
                    logger.warning(f"Could not identify scope for {raw_file.name}. Skipping.")
                    continue
                
                extracted_papers = self._extract_papers_from_events(events)
                papers_by_scope[scope].extend(extracted_papers)
                
            except Exception as e:
                logger.error(f"Error parsing raw response file {raw_file.name}: {e}")
        
        return papers_by_scope

    def _identify_scope(self, filename: str) -> Optional[str]:
        """Extracts scope (e.g., 'core_task', 'contribution_1') from filename."""
        match = re.match(r'raw_(core_task|contribution_\d+)_.*\.json', filename)
        if match:
            return match.group(1)
        return None

    def _extract_papers_from_events(self, events: List[Dict]) -> List[Dict]:
        """
        Extract paper metadata from Wispaper SSE events and generate quality flags.
        
        Parses verification events from Wispaper's streaming response, extracting
        paper metadata and calculating quality flags based on the verdict.
        
        Args:
            events: List of SSE event dictionaries from Wispaper API
            
        Returns:
            List of paper dictionaries with 'flags' field added
        """
        papers: List[Dict] = []
        for event in events:
            if event.get("event") == "onAgentEnd" and event.get("name") == "verification":
                data = event.get("data", {})
                metadata = data.get("metadata", {})
                content = data.get("content")  # This contains the verdict JSON string
                
                if not metadata:
                    continue

                # Parse verdict JSON from content
                verdict = None
                if isinstance(content, str):
                    try:
                        verdict = json.loads(content)
                    except json.JSONDecodeError:
                        logger.debug(f"Could not decode verdict JSON from content: {content[:100]}...")
                        verdict = None
                
                # Calculate quality flags using the extracted helper function
                flags = _calculate_quality_flags(verdict)
                
                paper = dict(metadata)  # Copy metadata
                paper['flags'] = flags  # Add generated flags
                papers.append(paper)
        
        return papers

    def _process_scope(
        self,
        scope: str,
        papers: List[Dict],
        topk_limit: int
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Applies filtering, deduplication, and TopK selection for a single scope.
        
        Returns:
            (topk_papers, stats_dict)
        """
        stats = {
            'scope': scope,
            'total': len(papers),
            'perfect': 0,
            'after_dedup': 0,
            'after_self_filter': 0,
            'after_temporal': 0,
            'topk': topk_limit,
            'final': 0,
        }
        
        # 1. Quality flag filtering (only 'perfect')
        perfect_papers = [p for p in papers if _is_perfect(p)]
        stats['perfect'] = len(perfect_papers)
        logger.debug(f"  Scope '{scope}': {len(perfect_papers)} perfect papers after quality filtering.")

        # 2. Attach canonical_id
        for paper in perfect_papers:
            self._attach_canonical_id(paper)
        
        # 3. Scope-internal deduplication (multi-layer: canonical_id + multi-ID + title)
        deduplicated_papers = self._deduplicate_multi_layer(perfect_papers)
        stats['after_dedup'] = len(deduplicated_papers)
        logger.debug(f"  Scope '{scope}': {len(deduplicated_papers)} papers after multi-layer deduplication.")

        # 4. Self-reference removal
        filtered_self = self._filter_self_reference(deduplicated_papers)
        stats['after_self_filter'] = len(filtered_self)
        logger.debug(f"  Scope '{scope}': {len(filtered_self)} papers after self-reference removal.")

        # 5. Temporal filtering
        filtered_temporal = self._filter_temporal(filtered_self)
        stats['after_temporal'] = len(filtered_temporal)
        logger.debug(f"  Scope '{scope}': {len(filtered_temporal)} papers after temporal filtering.")

        # 6. Rank by relevance score
        ranked_papers = sorted(filtered_temporal, key=_score, reverse=True)

        # 7. Top-K selection
        topk_papers = ranked_papers[:topk_limit]
        stats['final'] = len(topk_papers)
        
        return topk_papers, stats

    def _attach_canonical_id(self, paper: Dict):
        """
        Attaches a canonical_id to the paper dictionary.
        Handles the case where 'paper_id' might be a list.
        """
        # Fix: Handle paper_id being a list
        paper_id_field = paper.get('paper_id')
        if isinstance(paper_id_field, list):
            paper_id_field = paper_id_field[0] if paper_id_field else None
        elif paper_id_field is not None:
            paper_id_field = str(paper_id_field)  # Ensure it's a string if not list

        canonical_id = make_canonical_id(
            paper_id=paper_id_field,
            doi=paper.get('doi'),
            arxiv_id=None,  # Wispaper raw responses don't have a dedicated arxiv_id field
            url=paper.get('url'),
            pdf_url=paper.get('pdf_url'),
            title=paper.get('title'),
            year=paper.get('year'),
        )
        paper['canonical_id'] = canonical_id

    def _deduplicate_by_canonical_id(self, papers: List[Dict]) -> List[Dict]:
        """
        Deduplicates a list of papers based on canonical_id, keeping the one with the highest score.
        """
        unique_papers: Dict[str, Dict] = {}
        for paper in papers:
            cid = paper.get('canonical_id')
            if not cid:
                logger.warning(f"Paper missing canonical_id, skipping deduplication: {paper.get('title', 'N/A')}")
                continue
            
            current_score = _score(paper)
            if cid not in unique_papers or current_score > _score(unique_papers[cid]):
                unique_papers[cid] = paper
        
        return list(unique_papers.values())
    
    # Note: _extract_all_ids() and _deduplicate_by_any_id_match() have been removed.
    # They are no longer needed since canonical_id is now purely title-based.
    # The new deduplication strategy relies on:
    #   1. canonical_id (title hash) - primary dedup
    #   2. title dedup (fallback) - catches edge cases
    
    def _deduplicate_by_title(self, papers: List[Dict]) -> List[Dict]:
        """
        Final deduplication layer: merge papers with identical normalized titles.
        
        This catches edge cases where the same paper appears with completely 
        different IDs (e.g., one from OpenReview, another from ArXiv) that 
        canonical_id-based dedup missed.
        
        Priority for keeping duplicates:
            1. Has DOI (priority 3)
            2. Has ArXiv ID (priority 2)  
            3. Has OpenReview in canonical_id (priority 1)
            4. Others (priority 0)
            
        Ties broken by relevance score (higher wins).
        
        Args:
            papers: List of paper dicts after canonical_id dedup
            
        Returns:
            Further deduplicated list of papers
        """
        if len(papers) <= 1:
            return papers
        
        title_to_papers: Dict[str, List[Dict]] = {}
        for paper in papers:
            norm_title = _norm_title(paper.get('title', ''))
            if not norm_title:
                # Keep papers without titles
                title_to_papers[f"__notitle_{id(paper)}"] = [paper]
                continue
            
            if norm_title not in title_to_papers:
                title_to_papers[norm_title] = []
            title_to_papers[norm_title].append(paper)
        
        # From each title group, keep the one with highest priority
        result = []
        merged_count = 0
        for norm_title, group_papers in title_to_papers.items():
            if len(group_papers) > 1:
                # Priority: has DOI > has ArXiv > has OpenReview > others
                def priority(paper):
                    if paper.get('doi'):
                        return 3
                    if paper.get('arxiv_id'):
                        return 2
                    if 'openreview' in paper.get('canonical_id', ''):
                        return 1
                    return 0
                
                best = max(group_papers, key=lambda paper: (priority(paper), _score(paper)))
                result.append(best)
                merged_count += len(group_papers) - 1
            else:
                result.append(group_papers[0])
        
        if merged_count > 0:
            logger.debug(f"  Title dedup: merged {merged_count} duplicates, {len(papers)} -> {len(result)} papers")
        
        return result
    
    def _deduplicate_multi_layer(self, papers: List[Dict]) -> List[Dict]:
        """
        Apply two-layer deduplication strategy for scope-internal dedup.
        
        This is the primary deduplication within a single scope (e.g., core_task
        or contribution_1). Global cross-scope deduplication happens later in
        _generate_citation_index.
        
        Layers:
            1. canonical_id dedup (primary) - Handles 99.9% of cases since 
               canonical_id is now title-based MD5 hash
            2. Title dedup (fallback) - Catches edge cases where canonical_id 
               generation failed or produced different hashes for same title
        
        When duplicates are found, keeps the paper with:
            - Highest relevance score (within same priority)
            - Best ID quality: DOI > ArXiv > OpenReview > title-hash
        
        Args:
            papers: List of paper dicts, all with canonical_id attached
            
        Returns:
            Deduplicated list of papers
            
        Note:
            Multi-ID matching (arxiv_id, doi, openreview_id) was removed since
            canonical_id is now purely title-based, making it redundant.
        """
        original_count = len(papers)
        
        # Layer 1: canonical_id (covers 99.9% of cases)
        papers = self._deduplicate_by_canonical_id(papers)
        logger.debug(f"  Layer 1 (canonical_id): {original_count} -> {len(papers)} papers")
        
        # Layer 2: Title fallback (for edge cases)
        papers = self._deduplicate_by_title(papers)
        logger.debug(f"  Layer 2 (title): -> {len(papers)} papers")
        
        if len(papers) < original_count:
            logger.info(f"  Deduplication: {original_count} -> {len(papers)} papers "
                       f"(removed {original_count - len(papers)} duplicates)")
        
        return papers

    def _filter_self_reference(self, papers: List[Dict]) -> List[Dict]:
        """
        Removes the target paper itself from the candidates.
        
        Priority:
        1. canonical_id comparison (most reliable)
        2. PDF URL match (legacy support)
        3. Title match (legacy support)
        """
        # If we have no filtering info at all, return as-is
        if not self.original_paper_canonical_id and not self.self_title and not self.self_pdf_url:
            logger.warning("No original paper info available for self-filtering. Papers may include self-reference!")
            return papers

        filtered_papers = []
        removed_count = 0
        
        for paper in papers:
            is_self = False
            
            # Priority 1: Check canonical_id match (most reliable)
            if self.original_paper_canonical_id and paper.get('canonical_id'):
                if self.original_paper_canonical_id == paper.get('canonical_id'):
                    is_self = True
                    logger.debug(f"  Removed self-reference (canonical_id match): {paper.get('title', 'N/A')}")
            
            # Priority 2: Check PDF URL match (legacy support)
            if not is_self and self.self_pdf_url and paper.get('pdf_url'):
                if self._urls_match(self.self_pdf_url, paper['pdf_url']):
                    is_self = True
                    logger.debug(f"  Removed self-reference (URL match): {paper.get('title', 'N/A')}")
            
            # Priority 3: Check Title match (legacy support)
            if not is_self and self.self_title and paper.get('title'):
                if _norm_title(self.self_title) == _norm_title(paper['title']):
                    is_self = True
                    logger.debug(f"  Removed self-reference (title match): {paper.get('title', 'N/A')}")
            
            if not is_self:
                filtered_papers.append(paper)
            else:
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"  Removed {removed_count} self-reference(s) from candidates.")
        
        return filtered_papers

    def _urls_match(self, url1: str, url2: str) -> bool:
        """Helper to compare URLs robustly (e.g., ignoring http/https, www, trailing slashes)."""
        def normalize_url(url_str: str) -> str:
            if not url_str:
                return ""
            # Remove scheme, www, and trailing slash
            normalized = re.sub(r"^(https?://)?(www\.)?", "", url_str).rstrip('/')
            return normalized.lower()
        
        return normalize_url(url1) == normalize_url(url2)

    @staticmethod
    def _get_canonical_id_priority(cid: str) -> int:
        """
        Return canonical_id quality priority (higher = better).
        
        Priority order:
            4 - Real DOI (not arXiv DOI which uses 10.48550 prefix)
            3 - ArXiv ID
            2 - OpenReview ID  
            1 - Title-based hash (fallback)
            0 - Unknown/other format
        
        Args:
            cid: Canonical ID string (e.g., "doi:10.1234/...", "arxiv:2312.12345")
            
        Returns:
            Integer priority score (higher is better)
        """
        if cid.startswith("doi:") and ARXIV_DOI_PREFIX not in cid.lower():
            return CANONICAL_ID_PRIORITY["doi:"]
        elif cid.startswith("arxiv:"):
            return CANONICAL_ID_PRIORITY["arxiv:"]
        elif cid.startswith("openreview:"):
            return CANONICAL_ID_PRIORITY["openreview:"]
        elif cid.startswith("title:"):
            return CANONICAL_ID_PRIORITY["title:"]
        return 0

    def _filter_temporal(self, papers: List[Dict]) -> List[Dict]:
        """Removes papers published after the cutoff year."""
        if self.cutoff_year is None:
            return papers  # No cutoff year, skip filtering

        filtered_papers = []
        for paper in papers:
            paper_year = paper.get('year')
            if paper_year is None:
                filtered_papers.append(paper)  # Conservatively keep if year is missing
            elif isinstance(paper_year, (int, str)):
                try:
                    if int(paper_year) <= self.cutoff_year:
                        filtered_papers.append(paper)
                    else:
                        logger.debug(
                            f"  Removed temporal: {paper.get('title', 'N/A')} "
                            f"(year {paper_year} > cutoff {self.cutoff_year})"
                        )
                except ValueError:
                    filtered_papers.append(paper)  # Conservatively keep if year is invalid
            else:
                filtered_papers.append(paper)  # Conservatively keep if year is unexpected type
        
        return filtered_papers

    def _write_candidates_files(self, all_candidates: Dict[str, Dict[str, Any]]):
        """
        Generates candidates/ intermediate files.
        
        Format matches old phase2_searching output:
        - candidates/core_task_candidates.json
        - candidates/contributions/contribution_*.json
        """
        phase1_dir = self.phase2_dir.parent / "phase1"
        phase1_extracted_path = phase1_dir / "phase1_extracted.json"
        
        if not phase1_extracted_path.exists():
            logger.warning(f"Phase1 extracted file not found: {phase1_extracted_path}")
            return
        
        try:
            phase1_data = _read_json(phase1_extracted_path)
        except Exception as e:
            logger.error(f"Error reading phase1 extracted data: {e}")
            return
        
        # Core task candidates
        if SCOPE_CORE_TASK in all_candidates:
            core_task_info = phase1_data.get(SCOPE_CORE_TASK, {})
            core_task_text = core_task_info.get('text', '')
            queries = core_task_info.get('query_variants', [])
            
            # Get merged_dedup papers from TopK (before TopK selection)
            # We need to re-read raw_responses for this scope to get all papers
            core_papers = self._get_all_papers_for_scope(SCOPE_CORE_TASK)
            
            payload = {
                'core_task_text': core_task_text,
                'queries': queries,
                'merged_dedup': core_papers,
                'stats': all_candidates[SCOPE_CORE_TASK]
            }
            _write_json(self.candidates_dir / "core_task_candidates.json", payload)
            logger.info(f"Wrote candidates/core_task_candidates.json with {len(core_papers)} papers.")
        
        # Contribution candidates
        contributions = phase1_data.get('contributions', [])
        contrib_dir = self.candidates_dir / "contributions"
        contrib_dir.mkdir(exist_ok=True)
        
        for contrib in contributions:
            contrib_id = contrib.get('id', '')
            if contrib_id in all_candidates:
                queries = [contrib.get('prior_work_query', '')] + contrib.get('query_variants', [])
                queries = [q for q in queries if q]  # Remove empty strings
                
                # Get all papers for this contribution
                contrib_papers = self._get_all_papers_for_scope(contrib_id)
                
                payload = {
                    'contribution_id': contrib_id,
                    'contribution_name': contrib.get('name', ''),
                    'contribution_description': contrib.get('description', ''),
                    'prior_work_query': contrib.get('prior_work_query', ''),
                    'query_variants': contrib.get('query_variants', []),
                    'merged_dedup': contrib_papers,
                    'stats': all_candidates[contrib_id]
                }
                _write_json(contrib_dir / f"{contrib_id}.json", payload)
                logger.info(f"Wrote candidates/contributions/{contrib_id}.json with {len(contrib_papers)} papers.")

    def _get_all_papers_for_scope(self, scope: str) -> List[Dict]:
        """
        Re-parse raw_responses for a specific scope and return all filtered papers
        (perfect, dedup, self-filter, temporal) but before TopK selection.
        """
        papers = []
        for raw_file in self.raw_responses_dir.glob(f"raw_{scope}_*.json"):
            try:
                with open(raw_file, 'r', encoding='utf-8') as f:
                    events = json.load(f)
                papers.extend(self._extract_papers_from_events(events))
            except Exception as e:
                logger.error(f"Error reading {raw_file.name}: {e}")
        
        # Apply same filtering as _process_scope but without TopK
        perfect = [p for p in papers if _is_perfect(p)]
        for p in perfect:
            self._attach_canonical_id(p)
        deduped = self._deduplicate_by_canonical_id(perfect)
        filtered_self = self._filter_self_reference(deduped)
        filtered_temporal = self._filter_temporal(filtered_self)
        
        return filtered_temporal

    def _generate_citation_index(self, all_topk_papers: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Generate citation_index.json with global deduplication, roles, and LLM aliases.
        
        This is the Single Source of Truth for all papers in the analysis. It performs:
        1. Collects original paper (index 0) from phase1/paper.json
        2. Adds core_task papers (indices 1+) with their ranks
        3. Adds contribution papers with their scopes and ranks
        4. Deduplicates by normalized title (original paper always wins)
        5. Assigns LLM-generated or heuristic aliases for citation labels
        
        Output Structure:
            {
                "generated_at": "2026-01-18T12:00:00.000000+00:00",
                "count": 51,
                "items": [
                    {
                        "index": 0,
                        "canonical_id": "title:abc123...",
                        "alias": "BERT",
                        "title": "BERT: Pre-training...",
                        "roles": [{"type": "original_paper"}],
                        ...
                    },
                    {
                        "index": 1,
                        "canonical_id": "arxiv:2312.12345",
                        "alias": "Attention",
                        "title": "Attention Is All You Need",
                        "roles": [{"type": "core_task", "rank": 1}],
                        ...
                    },
                    ...
                ]
            }
        
        Deduplication Priority:
            1. Original paper (index 0) always wins over candidates with same title
            2. Among candidates: DOI > ArXiv > OpenReview > title-hash
        
        Args:
            all_topk_papers: Dict mapping scope names to lists of paper dicts
            
        Returns:
            Citation index dict ready for JSON serialization
        """
        by_canonical: Dict[str, Dict[str, Any]] = {}
        core_order: List[str] = []
        contrib_order: List[str] = []

        # Index 0: original paper
        phase1_dir = self.phase2_dir.parent / "phase1"
        original_paper_file = phase1_dir / PAPER_JSON
        
        if original_paper_file.exists():
            try:
                original_paper = _read_json(original_paper_file)
                
                # Always regenerate canonical_id to ensure it uses the current strategy
                # (title-based MD5 hash). Don't use stored canonical_id, which may be
                # from an older generation strategy (DOI/ArXiv/OpenReview priority).
                canonical_id = make_canonical_id(
                    paper_id=original_paper.get("paper_id"),
                    doi=original_paper.get("doi"),
                    arxiv_id=original_paper.get("arxiv_id"),
                    url=original_paper.get("url"),
                    pdf_url=original_paper.get("pdf_url"),
                    title=original_paper.get("title"),
                    year=original_paper.get("year"),
                )

                url = original_paper.get("url") or original_paper.get("original_pdf_url")
                if not url:
                    pid = original_paper.get("paper_id", "")
                    if isinstance(pid, str) and pid.startswith("http"):
                        url = pid

                # Extract openreview_id with proper formatting
                openreview_id = _ensure_openreview_id(original_paper)

                by_canonical[canonical_id] = {
                    "canonical_id": canonical_id,
                    "roles": [{"type": ROLE_ORIGINAL_PAPER}],
                    "title": original_paper.get("title", ""),
                    "authors": original_paper.get("authors", []),
                    "venue": original_paper.get("venue", ""),
                    "year": original_paper.get("year", 0),
                    "url": url or "",
                    "doi": _norm_doi(original_paper.get("doi")),
                    "arxiv_id": original_paper.get("arxiv_id") or _extract_arxiv_id_from_canonical(canonical_id),
                    "openreview_id": openreview_id,
                    "abstract": original_paper.get("abstract", ""),
                }
            except Exception as e:
                logger.warning(f"Error reading original paper: {e}")

        # Core task papers (indices 1+)
        if SCOPE_CORE_TASK in all_topk_papers:
            for rank_idx, paper in enumerate(all_topk_papers[SCOPE_CORE_TASK], start=1):
                # Regenerate canonical_id to ensure consistency
                try:
                    canonical_id = make_canonical_id(
                        paper_id=paper.get("paper_id"),
                        doi=paper.get("doi"),
                        arxiv_id=paper.get("arxiv_id"),
                        url=paper.get("url") or paper.get("source_url"),
                        pdf_url=paper.get("pdf_url"),
                        title=paper.get("title"),
                        year=paper.get("year"),
                    )
                except Exception:
                    canonical_id = paper.get("canonical_id", "")
                
                if not canonical_id:
                    continue
                
                entry = by_canonical.get(canonical_id)
                if not entry:
                    entry = {
                        "canonical_id": canonical_id,
                        "roles": [],
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "venue": paper.get("venue", ""),
                        "year": paper.get("year", 0),
                        "url": paper.get("url") or paper.get("pdf_url") or paper.get("source_url") or "",
                        "pdf_url": paper.get("pdf_url", ""),
                        "source_url": paper.get("source_url", ""),
                        "paper_id": paper.get("paper_id", ""),
                        "doi": _norm_doi(paper.get("doi")),
                        "arxiv_id": paper.get("arxiv_id") or _extract_arxiv_id_from_canonical(canonical_id),
                        "openreview_id": _ensure_openreview_id(paper),
                        "abstract": paper.get("abstract", ""),
                        "relevance_score": paper.get("relevance_score") or paper.get("final_score"),
                        "flags": paper.get("flags", {}),
                        "citations": paper.get("citations"),
                        "research_field": paper.get("research_field"),
                    }
                    by_canonical[canonical_id] = entry
                
                entry["roles"].append({"type": ROLE_CORE_TASK, "rank": rank_idx})
                core_order.append(canonical_id)

        # Contribution papers (after core task)
        for scope, papers in all_topk_papers.items():
            if scope == SCOPE_CORE_TASK:
                continue
            
            for rank, paper in enumerate(papers, start=1):
                # Regenerate canonical_id to ensure consistency
                try:
                    canonical_id = make_canonical_id(
                        paper_id=paper.get("paper_id"),
                        doi=paper.get("doi"),
                        arxiv_id=paper.get("arxiv_id"),
                        url=paper.get("url") or paper.get("source_url"),
                        pdf_url=paper.get("pdf_url"),
                        title=paper.get("title"),
                        year=paper.get("year"),
                    )
                except Exception:
                    canonical_id = paper.get("canonical_id", "")
                
                if not canonical_id:
                    continue
                
                entry = by_canonical.get(canonical_id)
                if not entry:
                    entry = {
                        "canonical_id": canonical_id,
                        "roles": [],
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", []),
                        "venue": paper.get("venue", ""),
                        "year": paper.get("year", 0),
                        "url": paper.get("url") or paper.get("pdf_url") or paper.get("source_url") or "",
                        "pdf_url": paper.get("pdf_url", ""),
                        "source_url": paper.get("source_url", ""),
                        "paper_id": paper.get("paper_id", ""),
                        "doi": _norm_doi(paper.get("doi")),
                        "arxiv_id": paper.get("arxiv_id") or _extract_arxiv_id_from_canonical(canonical_id),
                        "openreview_id": _ensure_openreview_id(paper),
                        "abstract": paper.get("abstract", ""),
                        "relevance_score": paper.get("relevance_score") or paper.get("final_score"),
                        "flags": paper.get("flags", {}),
                        "citations": paper.get("citations"),
                        "research_field": paper.get("research_field"),
                    }
                    by_canonical[canonical_id] = entry
                    contrib_order.append(canonical_id)
                
                # Add contribution role with scope and rank
                entry["roles"].append({"type": ROLE_CONTRIBUTION, "scope": scope, "rank": rank})

        # Secondary deduplication: merge papers with identical normalized title
        # (NOTE: Year is NOT used in key - same paper can have different years in different sources)
        # SPECIAL RULE: Original paper (index 0) always wins - delete candidates with same title
        title_map: Dict[str, str] = {}
        duplicates_to_remove: List[str] = []
        
        # Find original paper's canonical_id and title
        original_cid = None
        original_title = None
        for cid, entry in by_canonical.items():
            roles = entry.get("roles", [])
            if any(r.get("type") == ROLE_ORIGINAL_PAPER for r in roles):
                original_cid = cid
                original_title = entry.get("title", "").strip().lower()
                original_title = re.sub(r"[^\w\s]", " ", original_title)
                original_title = re.sub(r"\s+", " ", original_title).strip()
                break
        
        # Pass 1: Remove any candidates that match the original paper's title
        if original_cid and original_title:
            for canonical_id, entry in list(by_canonical.items()):
                if canonical_id == original_cid:
                    continue  # Skip original paper itself
                
                title = entry.get("title", "").strip().lower()
                title = re.sub(r"[^\w\s]", " ", title)
                title = re.sub(r"\s+", " ", title).strip()
                
                if title == original_title:
                    # This candidate has the same title as the original paper
                    # Remove the candidate, keep the original paper unchanged
                    duplicates_to_remove.append(canonical_id)
                    logger.info(
                        f"Removing candidate (same title as original): "
                        f"{canonical_id} -> keeping original {original_cid}"
                    )
        
        # Remove candidates that match original paper
        for cid in duplicates_to_remove:
            by_canonical.pop(cid, None)
            # Remove from order lists
            while cid in core_order:
                core_order.remove(cid)
            while cid in contrib_order:
                contrib_order.remove(cid)
        
        # Pass 2: Deduplicate among remaining candidates (not including original paper)
        duplicates_to_merge: Dict[str, str] = {}
        core_position = {cid: paper_idx for paper_idx, cid in enumerate(core_order)}
        
        # Use static method for ID priority calculation
        get_id_priority = self._get_canonical_id_priority
        
        for canonical_id, entry in list(by_canonical.items()):
            # Skip original paper in deduplication
            roles = entry.get("roles", [])
            if any(role.get("type") == ROLE_ORIGINAL_PAPER for role in roles):
                continue
            
            title = entry.get("title", "").strip().lower()
            title = re.sub(r"[^\w\s]", " ", title)
            title = re.sub(r"\s+", " ", title).strip()
            
            if not title:
                continue
            
            # Use only title for deduplication (not year)
            key = title
            if key in title_map:
                # Found duplicate among candidates - apply merge logic
                existing_cid = title_map[key]
                existing_entry = by_canonical[existing_cid]
                
                # Determine position and ID quality
                existing_in_core = existing_cid in core_position
                current_in_core = canonical_id in core_position
                
                better_position_is_existing = True
                if existing_in_core and current_in_core:
                    better_position_is_existing = (core_position[existing_cid] <= core_position[canonical_id])
                elif current_in_core and not existing_in_core:
                    better_position_is_existing = False
                
                existing_priority = get_id_priority(existing_cid)
                current_priority = get_id_priority(canonical_id)
                better_id_is_existing = (existing_priority >= current_priority)
                
                # Merge strategy
                if better_position_is_existing and better_id_is_existing:
                    duplicates_to_merge[canonical_id] = existing_cid
                    existing_entry["roles"].extend(entry.get("roles", []))
                elif better_position_is_existing and not better_id_is_existing:
                    duplicates_to_merge[existing_cid] = canonical_id
                    title_map[key] = canonical_id
                    entry["roles"] = existing_entry.get("roles", []) + entry.get("roles", [])
                    if not entry.get("abstract") and existing_entry.get("abstract"):
                        entry["abstract"] = existing_entry["abstract"]
                    by_canonical.pop(existing_cid, None)
                    by_canonical[canonical_id] = entry
                elif not better_position_is_existing and better_id_is_existing:
                    duplicates_to_merge[canonical_id] = existing_cid
                    existing_entry["roles"].extend(entry.get("roles", []))
                else:
                    duplicates_to_merge[existing_cid] = canonical_id
                    title_map[key] = canonical_id
                    entry["roles"].extend(existing_entry.get("roles", []))
                    if not entry.get("abstract") and existing_entry.get("abstract"):
                        entry["abstract"] = existing_entry["abstract"]
            else:
                title_map[key] = canonical_id
        
        # Remove duplicates
        for dup_cid in list(duplicates_to_merge.keys()):
            if dup_cid != duplicates_to_merge.get(dup_cid):
                by_canonical.pop(dup_cid, None)
        
        # Update references in order lists
        core_order = [duplicates_to_merge.get(k, k) for k in core_order]
        contrib_order = [duplicates_to_merge.get(k, k) for k in contrib_order]
        
        logger.info(
            f"Secondary deduplication: merged "
            f"{len([cid for cid, target_id in duplicates_to_merge.items() if cid != target_id])} duplicate papers"
        )
        
        # Assign indices: original (0) then core-task order then contribution first-seen
        ordered: List[str] = []
        orig = next(
            (cid for cid, paper_entry in by_canonical.items() 
             if any(role.get("type") == ROLE_ORIGINAL_PAPER for role in paper_entry.get("roles", []))),
            None
        )
        if orig:
            ordered.append(orig)
        for k in core_order:
            if k not in ordered:
                ordered.append(k)
        for k in contrib_order:
            if k not in ordered:
                ordered.append(k)

        # Build final list
        out: List[Dict[str, Any]] = []
        for index, cid in enumerate(ordered):
            entry = by_canonical[cid]
            out.append({
                "index": index,
                "canonical_id": cid,
                "alias": "",  # Will be filled by LLM
                "title": entry.get("title", ""),
                "authors": entry.get("authors", []),
                "venue": entry.get("venue", ""),
                "year": entry.get("year", 0),
                "url": entry.get("url", ""),
                "pdf_url": entry.get("pdf_url", ""),
                "source_url": entry.get("source_url", ""),
                "paper_id": entry.get("paper_id", ""),
                "doi": entry.get("doi"),
                "arxiv_id": entry.get("arxiv_id"),
                "openreview_id": entry.get("openreview_id"),
                "roles": entry.get("roles", []),
                "abstract": entry.get("abstract", ""),
                "relevance_score": entry.get("relevance_score"),
                "flags": entry.get("flags", {}),
                "citations": entry.get("citations"),
                "research_field": entry.get("research_field"),
            })

        # Assign aliases (LLM by default; heuristic fallback)
        self._assign_aliases_llm_or_fallback(out)

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec='microseconds'),
            "count": len(out),
            "items": out,
        }

    def _assign_aliases_llm_or_fallback(self, items: List[Dict[str, Any]]) -> None:
        """
        Assign `alias` for each item in-place using LLM or heuristic fallback.
        
        This function is adapted from old postprocess._assign_aliases_llm_or_fallback().
        """
        mode = (os.getenv("PHASE2_ALIAS_MODE") or "llm").strip().lower()

        # Collect targets needing aliases
        pending: List[Tuple[int, Dict[str, Any], str]] = []
        for item_idx, item in enumerate(items):
            if item.get("alias"):
                continue
            title = (item.get("title") or "").strip()
            pending.append((item_idx, item, title))

        if not pending:
            return

        def assign_heuristic(targets: List[Tuple[int, Dict[str, Any], str]]) -> None:
            for _, item, title in targets:
                item["alias"] = self._make_alias_from_title(title)

        if mode == "heuristic":
            assign_heuristic(pending)
            return

        # Try LLM
        llm_client = None
        try:
            from paper_novelty_pipeline.services.llm_client import create_llm_client
            llm_client = create_llm_client()
        except Exception as e:
            logger.warning(f"Phase2 alias: failed to init LLM client; falling back to heuristic. err={e}")
            llm_client = None

        if not llm_client:
            assign_heuristic(pending)
            return

        # Batched LLM call
        titles = [paper_title for _, _, paper_title in pending]
        system_prompt = (
            "You generate very short, human-readable aliases for research papers.\n"
            "Your goal is to create a memorable alias that can be used as a citation label.\n"
            "Rules:\n"
            "- Use 1 to 3 words max.\n"
            "- Prefer meaningful tokens from the TITLE.\n"
            "- Do NOT include the publication year.\n"
            "- Do NOT include numeric indices.\n"
            "- Do NOT include square brackets [] or parentheses ().\n"
            "- Output MUST be STRICT JSON only: {\"aliases\": [\"...\", ...]}.\n"
            "- The aliases array MUST have the same length and order as the input titles.\n"
        )
        user_payload = {"titles": titles}
        user_prompt = (
            "Given the following paper titles, generate one alias per title, in order.\n"
            "Return JSON of the form:\n"
            "{\"aliases\": [\"Alias for title_1\", \"Alias for title_2\", ...]}\n\n"
            + json.dumps(user_payload, ensure_ascii=True)
        )

        aliases: Optional[List[str]] = None
        try:
            result = None
            try:
                result = llm_client.generate_json(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=2000,
                    temperature=0.2,
                )
            except Exception:
                result = None

            if isinstance(result, dict):
                raw_aliases = result.get("aliases")
                if isinstance(raw_aliases, list) and all(isinstance(a, str) for a in raw_aliases):
                    if len(raw_aliases) == len(pending):
                        aliases = [a.strip() for a in raw_aliases]

            # Fallback: plain text generation + parse
            if aliases is None:
                raw = ""
                try:
                    raw = llm_client.generate(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=2000,
                        temperature=0.2,
                    )
                except Exception:
                    raw = ""

                if isinstance(raw, str) and raw.strip():
                    try:
                        parsed = json.loads(raw)
                        raw_aliases = parsed.get("aliases")
                        if isinstance(raw_aliases, list) and all(isinstance(a, str) for a in raw_aliases):
                            if len(raw_aliases) == len(pending):
                                aliases = [a.strip() for a in raw_aliases]
                    except Exception:
                        line_list = [line.strip() for line in raw.splitlines() if line.strip()]
                        if len(line_list) >= len(pending):
                            aliases = line_list[:len(pending)]
        except Exception as e:
            logger.warning(f"Phase2 alias: LLM alias generation failed; falling back. err={e}")
            aliases = None

        if aliases is None:
            assign_heuristic(pending)
            return

        # Sanitize + assign
        for (_, item, title), alias in zip(pending, aliases):
            if not isinstance(alias, str) or not alias.strip():
                item["alias"] = self._make_alias_from_title(title)
                continue
            cleaned = alias.strip().strip('"').strip("'")
            cleaned = re.sub(r"[\[\]()]", "", cleaned)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            item["alias"] = cleaned or self._make_alias_from_title(title)

    def _make_alias_from_title(self, title: str) -> str:
        """Heuristic alias generation from title (first 4 words)."""
        if not isinstance(title, str) or not title.strip():
            return "Paper"
        title_text = title.strip()
        if ":" in title_text:
            head = title_text.split(":", 1)[0].strip()
            if head:
                title_text = head
        words = title_text.split()
        alias = " ".join(words[:4])
        alias = re.sub(r"[^\w\s-]", "", alias)
        alias = re.sub(r"\s+", " ", alias).strip()
        return alias or "Paper"

    def _derive_topk_from_citation_index(self, citation_index: Dict[str, Any]) -> Dict[str, int]:
        """
        Derive TopK files from citation_index using the roles information.
        
        This ensures all output files use the same canonical_id from citation_index,
        maintaining consistency across all Phase2 outputs. The citation_index is the
        Single Source of Truth, and TopK files are derived views of it.
        
        Output Files:
            - final/core_task_perfect_top{N}.json - Top N core task candidates
            - final/contribution_1_perfect_top{M}.json - Top M for contribution 1
            - final/contribution_2_perfect_top{M}.json - etc.
        
        Note:
            If a paper has multiple ranks in the same scope (due to secondary 
            deduplication merging), we use the minimum rank to determine position.
        
        Args:
            citation_index: The generated citation index dict
            
        Returns:
            Dict mapping scope names to count of papers written
            
        Example:
            >>> result = processor._derive_topk_from_citation_index(citation_index)
            >>> result
            {'core_task': 50, 'contribution_1': 10, 'contribution_2': 8}
        """
        items = citation_index.get('items', [])
        
        # Group papers by scope, using minimum rank for papers with multiple ranks
        scopes: Dict[str, Dict[str, Tuple[int, Dict]]] = defaultdict(dict)
        
        for item in items:
            canonical_id = item['canonical_id']
            
            for role in item.get('roles', []):
                role_type = role.get('type')
                rank = role.get('rank')
                
                if role_type == ROLE_CORE_TASK and rank is not None:
                    # Use minimum rank if paper appears multiple times
                    if canonical_id not in scopes[SCOPE_CORE_TASK] or rank < scopes[SCOPE_CORE_TASK][canonical_id][0]:
                        scopes[SCOPE_CORE_TASK][canonical_id] = (rank, item)
                
                elif role_type == ROLE_CONTRIBUTION:
                    scope = role.get('scope')
                    contrib_rank = role.get('rank')
                    if scope and contrib_rank is not None:
                        # Use minimum rank if paper appears multiple times
                        if canonical_id not in scopes[scope] or contrib_rank < scopes[scope][canonical_id][0]:
                            scopes[scope][canonical_id] = (contrib_rank, item)
        
        # Write TopK files
        files_written = {}
        for scope, papers_dict in scopes.items():
            # Convert dict to list and sort by rank
            ranked_items = list(papers_dict.values())
            ranked_items.sort(key=lambda x: x[0])
            
            # Determine TopK limit
            topk_limit = self.topk_core_task if scope == SCOPE_CORE_TASK else self.topk_contribution
            
            # Take TopK
            topk_items = [item for rank, item in ranked_items[:topk_limit]]
            
            # Write file
            filename = f"{scope}_perfect_top{topk_limit}.json"
            filepath = self.final_dir / filename
            try:
                _write_json(filepath, topk_items)
                logger.info(f"Derived {len(topk_items)} papers to {filepath}")
                files_written[scope] = len(topk_items)
            except Exception as e:
                logger.error(f"Error writing derived TopK file {filepath}: {e}")
        
        return files_written

    def _write_citation_index(self, citation_index: Dict[str, Any]):
        """Writes the global citation_index.json."""
        filepath = self.final_dir / "citation_index.json"
        try:
            _write_json(filepath, citation_index)
            logger.info(f"Wrote citation_index to {filepath}")
        except Exception as e:
            logger.error(f"Error writing citation_index file {filepath}: {e}")

    def _write_metadata_files(
        self,
        citation_index: Dict[str, Any],
        all_candidates: Dict[str, Dict[str, Any]],
        derived_files: Dict[str, int]
    ):
        """
        Write metadata files for Phase3 compatibility:
        - contribution_mapping.json (required by comparison.py)
        - index.json (required as fallback)
        - contributions_index_top10.json (required as secondary fallback)
        - stats.json (helpful for debugging)
        """
        try:
            # 1. contribution_mapping.json
            self._write_contribution_mapping(citation_index)
            
            # 2. index.json
            self._write_index_json()
            
            # 3. contributions_index_top10.json
            self._write_contributions_index(all_candidates)
            
            # 4. stats.json
            self._write_stats_json(all_candidates, derived_files)
            
        except Exception as e:
            logger.error(f"Error writing metadata files: {e}")

    def _write_contribution_mapping(self, citation_index: Dict[str, Any]):
        """
        Generate contribution_mapping.json for Phase3 comparison.py.
        
        Maps each contribution to its TopK papers' canonical_ids.
        """
        contributions = []
        
        # Extract all contribution scopes from citation_index roles
        contribution_scopes = set()
        for item in citation_index.get('items', []):
            for role in item.get('roles', []):
                if role.get('type') == ROLE_CONTRIBUTION:
                    scope = role.get('scope')
                    if scope:
                        contribution_scopes.add(scope)
        
        # For each contribution scope, collect canonical_ids
        for scope in sorted(contribution_scopes):
            # Get all papers with this scope in their roles
            papers_in_scope = []
            for item in citation_index.get('items', []):
                for role in item.get('roles', []):
                    if role.get('type') == ROLE_CONTRIBUTION and role.get('scope') == scope:
                        rank = role.get('rank', 999)
                        papers_in_scope.append((rank, item['canonical_id']))
                        break
            
            # Sort by rank and extract canonical_ids
            papers_in_scope.sort(key=lambda x: x[0])
            canonical_ids = [cid for rank, cid in papers_in_scope]
            
            contributions.append({
                'contribution_id': scope,
                'contribution_name': '',  # Leave empty, can be filled later if needed
                'topk': self.topk_contribution,
                'papers_count': len(canonical_ids),
                'canonical_ids': canonical_ids
            })
        
        mapping = {
            'contributions': contributions,
            'metadata': {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'description': 'Mapping of contributions to their TopK candidate papers using canonical_id'
            }
        }
        
        filepath = self.final_dir / "contribution_mapping.json"
        _write_json(filepath, mapping)
        logger.info(f"Wrote contribution_mapping.json")

    def _write_index_json(self):
        """
        Generate index.json for Phase3 comparison.py fallback.
        
        Lists all TopK file paths (relative to phase2_dir for Phase3 compatibility).
        """
        # Use relative paths from phase2_dir (e.g., "final/contribution_1_perfect_top10.json")
        core_task_file = f"final/core_task_perfect_top{self.topk_core_task}.json"
        
        contribution_files = []
        for topk_file in sorted(self.final_dir.glob("contribution_*_perfect_top*.json")):
            contribution_files.append(f"final/{topk_file.name}")
        
        index = {
            'core_task_file': core_task_file,
            'contribution_files': contribution_files
        }
        
        filepath = self.final_dir / "index.json"
        _write_json(filepath, index)
        logger.info(f"Wrote index.json")

    def _write_contributions_index(self, all_candidates: Dict[str, Dict[str, Any]]):
        """
        Generate contributions_index_top10.json for Phase3 comparison.py.
        
        Lists stats for each contribution (using relative paths from phase2_dir).
        """
        contributions_index = []
        
        for scope, stats in sorted(all_candidates.items()):
            if scope == SCOPE_CORE_TASK:
                continue
            
            # Use relative paths from phase2_dir (e.g., "candidates/contributions/contribution_1.json")
            source_file = f"candidates/contributions/{scope}.json"
            output_file = f"final/{scope}_perfect_top{self.topk_contribution}.json"
            
            contributions_index.append({
                'id': scope,
                'source_file': source_file,
                'output_file': output_file,
                'stats': {
                    'total': stats.get('total', 0),
                    'perfect_total': stats.get('perfect', 0),
                    'unique_total': stats.get('after_dedup', 0),
                    'kept': stats.get('final', 0),
                    'topk': self.topk_contribution
                }
            })
        
        filepath = self.final_dir / "contributions_index_top10.json"
        _write_json(filepath, contributions_index)
        logger.info(f"Wrote contributions_index_top10.json")

    def _write_stats_json(self, all_candidates: Dict[str, Dict[str, Any]], derived_files: Dict[str, int]):
        """
        Generate stats.json for debugging and auditing.
        
        Contains filtering statistics for all scopes (using relative paths from phase2_dir).
        """
        core_stats = all_candidates.get(SCOPE_CORE_TASK, {})
        
        # Count total contributions stats
        contrib_stats = {
            'total_files': 0,
            'total_items': 0,
            'total_perfect': 0,
            'total_kept': 0
        }
        for scope, stats in all_candidates.items():
            if scope != SCOPE_CORE_TASK:
                contrib_stats['total_files'] += 1
                contrib_stats['total_items'] += stats.get('total', 0)
                contrib_stats['total_perfect'] += stats.get('perfect', 0)
                contrib_stats['total_kept'] += stats.get('final', 0)
        
        stats_data = {
            'core_task': {
                'source_file': "candidates/core_task_candidates.json",
                'output_file': f"final/core_task_perfect_top{self.topk_core_task}.json",
                'stats': {
                    'total': core_stats.get('total', 0),
                    'perfect_total': core_stats.get('perfect', 0),
                    'unique_total': core_stats.get('after_dedup', 0),
                    'kept': core_stats.get('final', 0),
                    'topk': self.topk_core_task
                }
            },
            'contributions': {
                'index_file': "final/contributions_index_top10.json",
                'summary': contrib_stats
            },
            'params': {
                'core_topk': self.topk_core_task,
                'contrib_topk': self.topk_contribution,
                'final_subdir': 'final'
            }
        }
        
        filepath = self.final_dir / "stats.json"
        _write_json(filepath, stats_data)
        logger.info(f"Wrote stats.json")


# ============================================================================
# Convenience Function
# ============================================================================

def postprocess_phase2_outputs(
    phase2_dir: Path,
    *,
    core_topk: int = 50,
    contrib_topk: int = 10,
    final_subdir: str = "final",
) -> Dict[str, Any]:
    """
    Post-process Phase2 outputs and write final TopK files + stats + citation_index.
    Returns the written stats dict (also persisted to final/stats.json).
    
    Compatible with the original postprocess_phase2_outputs interface.
    """
    processor = Phase2Processor(
        phase2_dir=phase2_dir,
        cutoff_year=None,
        self_pdf_url=None,
        self_title=None,
        topk_core_task=core_topk,
        topk_contribution=contrib_topk,
    )
    return processor.process()
