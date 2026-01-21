"""
Phase 3: Comparison Workflow

Core workflow for comparing original paper contributions against candidate papers.
Coordinates LLM analysis, evidence verification, and result aggregation.
"""

import os
import json
import logging
import re
import dataclasses
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from paper_novelty_pipeline.models import (
    PaperInput,
    ExtractedContent,
    SearchResult,
    RetrievedPaper,
    ComparisonResult,
    ContributionClaim,
    CoreTask,
    SearchQuery,
    ASPECT,
)
from paper_novelty_pipeline.utils.paths import PAPER_JSON, PHASE1_EXTRACTED_JSON

from paper_novelty_pipeline.phases.phase3.llm_analyzer import LLMAnalyzer
from paper_novelty_pipeline.phases.phase3.evidence_verifier import EvidenceVerifier
from paper_novelty_pipeline.phases.phase3.citation_manager import CitationManager
from paper_novelty_pipeline.phases.phase3.textual_similarity_detector import TextualSimilarityDetector
from paper_novelty_pipeline.phases.phase3.utils import (
    read_json_list,
    make_retrieved_paper,
    sanitize_id,
    normalize_title,
)
from paper_novelty_pipeline.phases.phase3.pdf_handler import PDFHandler
from paper_novelty_pipeline.config import (
    PHASE3_TOP_K,
    PHASE3_CONCURRENCY,
    LLM_MAX_PROMPT_CHARS,
    MAX_CONTEXT_CHARS,
    PHASE3_MIN_CONTEXT_CHARS,
    PROJECT_ROOT,
)


# ============================================================================
# Comparison Workflow Class
# ============================================================================

class ComparisonWorkflow:
    """
    Core workflow for contribution-level novelty comparison.
    
    Coordinates the entire comparison process:
    1. Loads Phase1/Phase2 data
    2. Prepares original and candidate texts
    3. Runs LLM-based analysis
    4. Verifies evidence pairs
    5. Aggregates and returns results
    """
    
    def __init__(
        self,
        llm_analyzer: LLMAnalyzer,
        evidence_verifier: EvidenceVerifier,
        pdf_handler: PDFHandler,
        logger: logging.Logger,
        output_base_dir: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize comparison workflow.
        
        Args:
            llm_analyzer: LLM analyzer instance for contribution analysis
            evidence_verifier: Evidence verifier instance for quote validation
            pdf_handler: PDF handler for text extraction
            logger: Logger instance
            output_base_dir: Base directory for outputs
            top_k: Maximum number of candidates to compare (None = all)
        """
        self.llm_analyzer = llm_analyzer
        self.evidence_verifier = evidence_verifier
        self.pdf_handler = pdf_handler
        self.logger = logger
        self.output_base_dir = output_base_dir
        self.top_k = top_k if top_k is not None else PHASE3_TOP_K
        
        # Cache for original full text to avoid repeated extraction
        self._orig_text_cache: Dict[str, Optional[str]] = {}
        
        # Initialize CitationManager
        base_dir_path = Path(output_base_dir) if output_base_dir else Path(".")
        self.citation_manager = CitationManager(
            logger=logger,
            base_dir=base_dir_path
        )
        # Try to load citation_index from survey_report
        self.citation_manager.load_from_survey_report()
    
    def compare_contributions_from_final(
        self,
        *,
        phase1_dir: Path,
        phase2_dir: Path,
        out_dir: Path,
        language: str = "en",
        max_candidates: int = 10,
        contribution_indices: Optional[List[int]] = None,
        resume: bool = False,
    ) -> List[ComparisonResult]:
        """
        **MAIN ENTRY POINT** for Phase3 contribution novelty comparison.
        
        Compares the original paper's contributions (from Phase1) against top-K
        candidate papers (from Phase2) to assess novelty.
        
        Workflow:
        1. Load Phase1 extracted contributions (1-3 items)
        2. Load Phase2 final TopK candidates per contribution (default: top 10)
        3. For each contribution, compare against its candidates:
           - Download/extract full text from both papers
           - Run LLM-based novelty analysis
           - Verify evidence pairs (fuzzy quote matching)
           - Detect textual similarity segments
        4. Generate per-contribution novelty reports (JSON)
        5. Write contribution-centric projection summary
        
        Args:
            phase1_dir: Directory containing phase1_extracted.json
            phase2_dir: Directory containing final/contribution_X_topk.json files
            out_dir: Output directory for Phase3 reports
            language: Language for reports (default: "en")
            max_candidates: Max candidates to compare per contribution (default: 10)
            contribution_indices: Optional list of contribution indices to process (0-based)
            resume: If True, skip already processed comparisons
        
        Returns:
            List of ComparisonResult objects (one per contribution-candidate pair)
        """
        self.output_base_dir = str(out_dir)
        # Ensure we don't truncate by global top_k
        self.top_k = None
        
        # Load citation index for Contribution Analysis (Phase2 canonical source)
        citation_index_path = Path(phase2_dir) / "final" / "citation_index.json"
        original_canonical_id = None
        if citation_index_path.exists():
            self.logger.info(f"Loading citation index from {citation_index_path}")
            try:
                # Read canonical_id from Phase2 citation_index.json (index 0 = original paper)
                citation_data = json.loads(citation_index_path.read_text(encoding="utf-8"))
                items = citation_data.get("items", [])
                if items and len(items) > 0:
                    original_item = items[0]
                    roles = original_item.get("roles", [])
                    if any(r.get("type") == "original_paper" for r in roles):
                        original_canonical_id = original_item.get("canonical_id")
                        self.logger.info(f"Phase3: Loaded original canonical_id from Phase2: {original_canonical_id}")
                
                # Reuse unified loader so that keys are canonical_id
                if not self.citation_manager.load_from_phase2_citation(citation_index_path):
                    self.logger.warning("Failed to initialize citation index from Phase2.")
                else:
                    self.logger.info(
                        f"Loaded {len(self.citation_manager.citation_index)} papers into citation index"
                    )
            except Exception as e:
                self.logger.warning(f"Failed to load citation index: {e}")
        
        # Load Phase1 data
        p1p = Path(phase1_dir) / PAPER_JSON
        p1e = Path(phase1_dir) / PHASE1_EXTRACTED_JSON
        paper = json.loads(p1p.read_text(encoding="utf-8"))
        extracted = json.loads(p1e.read_text(encoding="utf-8"))
        
        original = PaperInput(
            paper_id=str(paper.get("canonical_id") or paper.get("paper_id") or ""),
            title=paper.get("title") or "",
            abstract=paper.get("abstract") or "",
            authors=paper.get("authors") or [],
            venue=paper.get("venue") or "",
            year=paper.get("year"),
            doi=paper.get("doi"),
            arxiv_id=paper.get("arxiv_id"),
            original_pdf_url=paper.get("original_pdf_url"),
            # canonical_id is not a field in PaperInput
            openreview_forum_id=paper.get("openreview_forum_id"),
            openreview_pdf_path=paper.get("openreview_pdf_path"),
            keywords=paper.get("keywords"),
            primary_area=paper.get("primary_area"),
            openreview_rating_mean=paper.get("openreview_rating_mean"),
        )
        
        contribs: List[Dict[str, Any]] = list(extracted.get("contributions") or [])
        if isinstance(contribution_indices, list) and contribution_indices:
            contribs = [contribs[i] for i in contribution_indices if 0 <= i < len(contribs)]
        
        # Build ExtractedContent using the new structure (contributions field)
        core_task_data = extracted.get("core_task", {})
        core_task = CoreTask(
            text=core_task_data.get("text", "") if isinstance(core_task_data, dict) else "",
            query_variants=core_task_data.get("query_variants", []) if isinstance(core_task_data, dict) else [],
        )
        
        # Build ContributionClaim objects directly from JSON data
        contribution_claims: List[ContributionClaim] = []
        for c in contribs:
            contribution_claims.append(
                ContributionClaim(
                    id=c.get("id") or f"contribution_{len(contribution_claims)}",
                    name=c.get("name") or "contribution",
                    author_claim_text=c.get("author_claim_text") or "",
                    description=c.get("description") or c.get("author_claim_text") or "",
                    prior_work_query=c.get("prior_work_query") or "",
                    query_variants=c.get("query_variants", []),
                    source_hint=c.get("source_hint") or "unknown",
                )
            )
        
        # Create ExtractedContent with new structure (no slots needed)
        extracted_content = ExtractedContent(
            core_task=core_task,
            core_task_survey=None,
            contributions=contribution_claims,
        )
        
        # Load per-contribution TopK files
        search_results = self._load_phase2_results(phase2_dir, contribs, max_candidates, contribution_indices, extracted_content)
        
        # Extend citation_index with Contribution Analysis candidates
        self._extend_citation_index_for_contribution_analysis(search_results)

        # Run comparison and project results
        results = self.compare_papers(original, extracted_content, search_results, original_canonical_id=original_canonical_id)
        
        # Write contribution projection (if report generator is available)
        try:
            self._write_contribution_projection(out_dir, contribs, results)
        except Exception as e:
            self.logger.warning(f"Phase3: projection to contribution_analysis failed: {e}")
        
        return results

    def _classify_siblings(
        self,
        candidates: List[RetrievedPaper],
        original_paper_id: str,
        taxonomy_mapping: List[Dict[str, Any]],
    ) -> Tuple[List[RetrievedPaper], List[RetrievedPaper]]:
        """
        Classify candidates into siblings (same leaf node) and non-siblings.
        
        Args:
            candidates: List of candidate papers
            original_paper_id: Original paper ID (canonical_id from survey)
            taxonomy_mapping: List of {"paper_id": str, "taxonomy_path": List[str]}
            
        Returns:
            (siblings, non_siblings)
        """
        # Build canonical_id -> taxonomy_path mapping
        paper_to_path: Dict[str, List[str]] = {}
        for item in taxonomy_mapping:
            pid = item.get("canonical_id")
            path = item.get("taxonomy_path", [])
            if pid and path:
                paper_to_path[pid] = path
        
        original_path = paper_to_path.get(original_paper_id, [])
        
        if not original_path:
            self.logger.warning(
                f"Original paper {original_paper_id} not found in taxonomy mapping. "
                "All candidates will be treated as non-siblings."
            )
            return [], candidates
        
        siblings = []
        non_siblings = []
        
        for candidate in candidates:
            # Use canonical_id for matching (consistent with survey)
            cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
            candidate_path = paper_to_path.get(cid, [])
            
            # Same leaf = sibling
            if candidate_path and candidate_path == original_path:
                siblings.append(candidate)
            else:
                non_siblings.append(candidate)
        
        self.logger.info(
            f"Taxonomy classification: {len(siblings)} siblings (same leaf), "
            f"{len(non_siblings)} non-siblings"
        )
        
        return siblings, non_siblings
    
    def _compare_sibling_fulltext(
        self,
        candidate: RetrievedPaper,
        rank: int,
        original_paper: PaperInput,
        core_task_text: str,
        original_abstract: str,
        orig_full_text: Optional[str],
        taxonomy_context: Optional[Dict[str, Any]],
        comparisons_dir: Path,
        resume: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare sibling candidate using full-text extraction and detailed analysis.
        
        Similar to contribution analysis workflow:
        - Download PDF and extract full text
        - Run detailed LLM comparison
        - Verify evidence pairs
        
        Returns:
            Comparison result dict or None if failed
        """
        from datetime import datetime
        
        try:
            cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
            
            # Check if already processed (resume mode)
            if resume:
                safe_title = sanitize_id(candidate.title)[:50]
                existing_file = comparisons_dir / f"{rank:02d}_{safe_title}.json"
                if existing_file.exists():
                    self.logger.info(f"Skipping already processed sibling: {cid}")
                    with open(existing_file, "r", encoding="utf-8") as f:
                        return json.load(f)
            
            self.logger.info(f"Processing sibling {rank} (fulltext): {cid}")
            
            # Get candidate full text
            cand_text, cand_url, comparison_mode, skip_reason = self._prepare_candidate_text(candidate)
            
            if skip_reason and not cand_text:
                self.logger.warning(
                    f"Failed to get fulltext for sibling {cid}: {skip_reason}. "
                    "Falling back to abstract comparison."
                )
                # Fallback to abstract
                return self._compare_candidate_abstract(
                    candidate, rank, original_paper.title, original_abstract,
                    core_task_text, taxonomy_context, comparisons_dir,
                    relationship="sibling", fallback=True
                )
            
            # Normalize texts for comparison
            orig_text_normalized, cand_text_normalized = self._prepare_texts_for_comparison(
                orig_full_text or "", cand_text or ""
            )
            
            # Run LLM analysis (detailed for siblings)
            analysis_result = self.llm_analyzer.analyze_core_task_distinction(
                core_task_text=core_task_text,
                original_title=original_paper.title,
                original_abstract=original_abstract,
                original_paper_id=getattr(original_paper, "canonical_id", None) or original_paper.paper_id,
                candidate_title=candidate.title,
                candidate_abstract=candidate.abstract or "",
                candidate_paper_id=cid,
                taxonomy_context=taxonomy_context,
                relationship="sibling",
            )
            
            if not analysis_result:
                self.logger.warning(f"LLM analysis failed for sibling {cid}")
                return None
            
            # Build comparison result
            comparison_result: Dict[str, Any] = {
                "rank": rank,
                "relationship": "sibling",
                "comparison_mode": "fulltext",
                "candidate_paper_title": candidate.title,
                "candidate_paper_authors": candidate.authors or [],
                "candidate_paper_url": cand_url or candidate.source_url or "",
                "candidate_paper_abstract": candidate.abstract or "",
                "candidate_paper_venue": candidate.venue or "",
                "candidate_paper_year": candidate.year or 0,
                "retrieved_paper_id": cid,
                "relevance_score": candidate.relevance_score,
                "similarities": analysis_result.get("similarities", ""),
                "differences": analysis_result.get("differences", ""),
                "distinction_summary": analysis_result.get("differences", "").strip(),
                "original_paper_title": original_paper.title,
                "original_paper_abstract": original_abstract,
                "has_taxonomy_context": taxonomy_context is not None,
                "processed_at": datetime.now().isoformat(),
            }
            
            # Attach textual similarity segments if provided by analyzer
            segments = analysis_result.get("textual_similarity_segments") or []
            normalized_segments: List[Dict[str, Any]] = []
            for seg in segments:
                if isinstance(seg, dict):
                    normalized_segments.append(seg)
                else:
                    try:
                        normalized_segments.append(dataclasses.asdict(seg))
                    except Exception:
                        continue
            comparison_result["textual_similarity_segments"] = normalized_segments
            
            # Save individual file
            safe_title = sanitize_id(candidate.title)[:50]
            individual_file = comparisons_dir / f"{rank:02d}_{safe_title}.json"
            with open(individual_file, "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Completed sibling fulltext comparison: {cid}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Error in sibling fulltext comparison for {candidate.paper_id}: {e}", exc_info=True)
            return None
    
    def _compare_candidate_abstract(
        self,
        candidate: RetrievedPaper,
        rank: int,
        original_title: str,
        original_abstract: str,
        core_task_text: str,
        taxonomy_context: Optional[Dict[str, Any]],
        comparisons_dir: Path,
        relationship: str = "non-sibling",
        fallback: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare non-sibling candidate using abstract-only comparison (1-2 sentences).
        
        Args:
            candidate: Candidate paper
            rank: Rank in the list
            original_title: Original paper title
            original_abstract: Original paper abstract
            core_task_text: Core task text
            taxonomy_context: Optional taxonomy context
            comparisons_dir: Output directory
            relationship: Relationship type (default: "non-sibling")
            fallback: Whether this is a fallback from fulltext failure
            
        Returns:
            Comparison result dict or None if failed
        """
        from datetime import datetime
        
        try:
            cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
            
            self.logger.info(f"Processing {relationship} {rank} (abstract): {cid}")
            
            # Run LLM analysis (abstract-only fallback for siblings when fulltext unavailable)
            analysis_result = self.llm_analyzer.analyze_core_task_distinction(
                core_task_text=core_task_text,
                original_title=original_title,
                original_abstract=original_abstract,
                original_paper_id="",  # Not needed for brief comparison
                candidate_title=candidate.title,
                candidate_abstract=candidate.abstract or "",
                candidate_paper_id=cid,
                taxonomy_context=taxonomy_context,
                relationship=relationship,  # Pass through the relationship (should be "sibling")
            )
            
            if not analysis_result:
                self.logger.warning(f"LLM analysis failed for {relationship} {cid}")
                return None
            
            # For non-siblings, we only get a brief note
            brief_note = analysis_result.get("differences", "")
            
            comparison_result = {
                "rank": rank,
                "relationship": relationship,
                "comparison_mode": "abstract" if not fallback else "abstract_fallback",
                "candidate_paper_title": candidate.title,
                "candidate_paper_authors": candidate.authors or [],
                "candidate_paper_url": candidate.source_url or "",
                "candidate_paper_abstract": candidate.abstract or "",
                "candidate_paper_venue": candidate.venue or "",
                "candidate_paper_year": candidate.year or 0,
                "retrieved_paper_id": cid,
                "relevance_score": candidate.relevance_score,
                "brief_note": brief_note,
                "similarities": "",
                "differences": brief_note,
                "distinction_summary": brief_note,
                "original_paper_title": original_title,
                "original_paper_abstract": original_abstract,
                "has_taxonomy_context": taxonomy_context is not None,
                "processed_at": datetime.now().isoformat(),
            }
            
            # Save individual file
            safe_title = sanitize_id(candidate.title)[:50]
            individual_file = comparisons_dir / f"{rank:02d}_{safe_title}.json"
            with open(individual_file, "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Completed {relationship} abstract comparison: {cid}")
            return comparison_result
            
        except Exception as e:
            self.logger.error(f"Error in abstract comparison for {candidate.paper_id}: {e}", exc_info=True)
            return None
    
    def compare_core_task(
        self,
        *,
        phase1_dir: Path,
        phase2_dir: Path,
        out_dir: Path,
        max_candidates: Optional[int] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Compare original paper against top-K core_task candidates using abstract-level comparison.
        
        This is the Phase3 Core Task Analysis entry point.
        
        Workflow:
        1. Load Phase1 original paper info (title, abstract, core_task text, canonical_id)
        2. Load Phase2 final core_task_perfect_top50.json (TopK core-task candidates)
        3. (Best-effort) load taxonomy + mapping from Phase3 survey outputs
        4. For each candidate:
           - Check if candidate is a sibling (same taxonomy leaf) using taxonomy info
           - Call LLMAnalyzer.analyze_core_task_distinction for sibling papers only
           - Save individual comparison JSON under phase3/core_task_comparisons/
        5. Generate merged comparisons file + summary.json
        
        Returns:
            Dictionary with "statistics" describing comparison coverage.
        """
        from datetime import datetime

        self.output_base_dir = str(out_dir)

        # ------------------------------------------------------------------ #
        # Load Phase1 original paper info
        # ------------------------------------------------------------------ #
        p1p = Path(phase1_dir) / PAPER_JSON
        p1e = Path(phase1_dir) / PHASE1_EXTRACTED_JSON

        if not p1p.exists():
            raise FileNotFoundError(f"Phase1 {PAPER_JSON} not found: {p1p}")
        if not p1e.exists():
            raise FileNotFoundError(f"Phase1 {PHASE1_EXTRACTED_JSON} not found: {p1e}")

        paper_data = json.loads(p1p.read_text(encoding="utf-8"))
        extracted_data = json.loads(p1e.read_text(encoding="utf-8"))

        original_title = paper_data.get("title", "") or ""
        original_abstract = (
            paper_data.get("abstract", "")
            or (extracted_data.get("core_task", {}) or {}).get("description", "")
            or ""
        )
        original_paper_id = str(paper_data.get("paper_id") or "")
        
        # Read canonical_id from Phase2 citation_index.json (index 0 = original paper)
        original_canonical_id = None
        citation_index_path = Path(phase2_dir) / "final" / "citation_index.json"
        if citation_index_path.exists():
            try:
                citation_data = json.loads(citation_index_path.read_text(encoding="utf-8"))
                items = citation_data.get("items", [])
                if items and len(items) > 0:
                    original_item = items[0]
                    roles = original_item.get("roles", [])
                    if any(r.get("type") == "original_paper" for r in roles):
                        original_canonical_id = original_item.get("canonical_id")
                        self.logger.info(f"Phase3: Loaded original canonical_id from Phase2: {original_canonical_id}")
            except Exception as e:
                self.logger.warning(f"Phase3: Failed to load canonical_id from citation_index: {e}")
        
        # Fallback to paper_id if Phase2 not available
        if not original_canonical_id:
            original_canonical_id = original_paper_id
            self.logger.warning(f"Phase3: Using paper_id as canonical_id fallback: {original_canonical_id}")
        
        core_task_text = (extracted_data.get("core_task", {}) or {}).get("text", "") or ""

        # ------------------------------------------------------------------ #
        # Load Phase2 core_task candidates
        # ------------------------------------------------------------------ #
        core_task_file = Path(phase2_dir) / "final" / "core_task_perfect_top50.json"
        if not core_task_file.exists():
            raise FileNotFoundError(f"Core task candidates file not found: {core_task_file}")

        candidate_items = read_json_list(core_task_file)
        if not candidate_items:
            self.logger.warning(
                "Phase3: no candidate papers found in core_task_perfect_top50.json"
            )
            return {
                "statistics": {
                    "total_candidates": 0,
                    "eligible_candidates": 0,
                    "attempted": 0,
                    "completed": 0,
                    "completed_with_fallback": 0,
                    "errored": 0,
                    "skipped_by_design": 0,
                    "skip_reasons": {},
                    "taxonomy_status": "missing",
                }
            }

        # Limit candidates if specified
        if max_candidates and max_candidates > 0:
            candidate_items = candidate_items[: max_candidates]

        # Convert to RetrievedPaper objects (with canonical_id support)
        candidates: List[RetrievedPaper] = []
        for rank, item in enumerate(candidate_items, start=1):
            rp = make_retrieved_paper(item, fallback_score=rank)
            if rp:
                candidates.append(rp)

        if not candidates:
            self.logger.warning("Phase3: no valid candidates after conversion")
            return {
                "statistics": {
                    "total_candidates": 0,
                    "eligible_candidates": 0,
                    "attempted": 0,
                    "completed": 0,
                    "completed_with_fallback": 0,
                    "errored": 0,
                    "skipped_by_design": 0,
                    "skip_reasons": {},
                    "taxonomy_status": "missing",
                }
            }

        # ------------------------------------------------------------------ #
        # Output directory
        # ------------------------------------------------------------------ #
        base_dir = Path(out_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
        comparisons_dir = base_dir / "phase3" / "core_task_comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------ #
        # Strict dependency: taxonomy + survey_report are REQUIRED
        # ------------------------------------------------------------------ #
        survey_dir = base_dir / "phase3" / "core_task_survey"
        taxonomy_file = survey_dir / "taxonomy.json"
        survey_report_file = survey_dir / "survey_report.json"

        total_candidates = len(candidates)
        taxonomy_status: str = "missing"

        def _write_empty_outputs(
            *,
            position_type: str,
            note_en: str,
            skip_reason: str,
            taxonomy_path: Optional[List[str]] = None,
            sibling_candidate_count: int = 0,
            sibling_subtopic_count: int = 0,
            used_subtopic_level: bool = False,
            subtopic_comparison_file: Optional[str] = None,
        ) -> Dict[str, Any]:
            """Write empty comparisons + summary with clear 'skipped_by_design' semantics."""
            merged_file = comparisons_dir / "core_task_comparisons.json"
            with open(merged_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"comparisons": [], "metadata": {"total_comparisons": 0, "generated_at": datetime.now().isoformat()}},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            summary: Dict[str, Any] = {
                "statistics": {
                    "total_candidates": total_candidates,
                    "eligible_candidates": 0,
                    "attempted": 0,
                    "completed": 0,
                    "completed_with_fallback": 0,
                    "errored": 0,
                    "skipped_by_design": total_candidates,
                    "skip_reasons": {skip_reason: total_candidates},
                    "taxonomy_status": taxonomy_status,
                    "sibling_candidate_count": sibling_candidate_count,
                    "sibling_subtopic_count": sibling_subtopic_count,
                    "used_subtopic_level": bool(used_subtopic_level),
                },
                "structural_position": {
                    "taxonomy_path": taxonomy_path or [],
                    "position_type": position_type,
                    "sibling_candidate_count": sibling_candidate_count,
                    "sibling_subtopic_count": sibling_subtopic_count,
                    "note_en": note_en,
                },
                "generated_at": datetime.now().isoformat(),
            }
            if subtopic_comparison_file:
                summary["subtopic_comparison_file"] = subtopic_comparison_file

            with open(comparisons_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            return summary

        if not (taxonomy_file.exists() and survey_report_file.exists()):
            taxonomy_status = "missing"
            return _write_empty_outputs(
                position_type="taxonomy_unavailable",
                note_en=(
                    "Taxonomy is unavailable (missing phase3/core_task_survey/taxonomy.json or survey_report.json). "
                    "Core-task comparisons were skipped."
                ),
                skip_reason="taxonomy_unavailable",
            )

        # Load taxonomy + mapping (required)
        try:
            taxonomy = json.loads(taxonomy_file.read_text(encoding="utf-8"))
            survey_report = json.loads(survey_report_file.read_text(encoding="utf-8"))
        except Exception as e:
            taxonomy_status = "missing"
            return _write_empty_outputs(
                position_type="taxonomy_unavailable",
                note_en=f"Failed to load taxonomy/survey_report JSON: {e}. Core-task comparisons were skipped.",
                skip_reason="taxonomy_unavailable",
            )

        short = survey_report.get("short_survey", {}) or {}
        taxonomy_mapping: List[Dict[str, Any]] = short.get("mapping", []) or []
        highlight = short.get("highlight", {}) or {}
        highlight_id = highlight.get("original_paper_id")
        taxonomy_original_id: str = original_canonical_id
        if isinstance(highlight_id, str) and highlight_id.strip():
            taxonomy_original_id = highlight_id.strip()

        diag = short.get("taxonomy_diagnostics") or {}
        st = diag.get("status")
        taxonomy_status = st.strip() if isinstance(st, str) and st.strip() else "missing"

        # Build canonical_id -> taxonomy_path
        paper_to_path: Dict[str, List[str]] = {}
        for item in taxonomy_mapping:
            pid = item.get("canonical_id")
            path = item.get("taxonomy_path")
            if isinstance(pid, str) and isinstance(path, list):
                clean = [x.strip() for x in path if isinstance(x, str) and x.strip()]
                if clean:
                    paper_to_path[pid.strip()] = clean

        orig_path = paper_to_path.get(taxonomy_original_id) or []
        if not orig_path:
            return _write_empty_outputs(
                position_type="taxonomy_unavailable",
                note_en=(
                    "Taxonomy mapping does not contain the original paper; core-task comparisons were skipped."
                ),
                skip_reason="taxonomy_mapping_missing_original",
            )

        parent_path = orig_path[:-1]
        leaf_name = orig_path[-1]

        # candidate id -> rank/title/abstract
        candidate_meta: Dict[str, Dict[str, Any]] = {}
        for r, cand in enumerate(candidates, start=1):
            cid = getattr(cand, "canonical_id", None) or cand.paper_id
            candidate_meta[cid] = {
                "rank": r,
                "title": getattr(cand, "title", "") or "",
                "abstract": getattr(cand, "abstract", "") or "",
            }

        # Sibling candidates = same leaf AND in phase2 TopK AND not original
        sibling_candidate_ids: List[str] = [
            pid
            for pid, path in paper_to_path.items()
            if path == orig_path and pid != taxonomy_original_id and pid in candidate_meta
        ]
        sibling_candidate_count = len(sibling_candidate_ids)

        # Sibling subtopics = other leaf names under same parent
        sibling_subtopic_names = sorted(
            {
                path[-1]
                for pid, path in paper_to_path.items()
                if pid != taxonomy_original_id and path[:-1] == parent_path and len(path) == len(orig_path)
            }
            - {leaf_name}
        )
        sibling_subtopic_count = len(sibling_subtopic_names)

        def _find_node_by_path(taxo_root: Dict[str, Any], path: List[str]) -> Optional[Dict[str, Any]]:
            """Locate a taxonomy node by its path of names."""
            if not path:
                return None
            current = taxo_root
            idx = 0
            root_name = taxo_root.get("name")
            if root_name and path and path[0] == root_name:
                idx = 1
            for name in path[idx:]:
                found = None
                for st in current.get("subtopics", []) or []:
                    if st.get("name") == name:
                        found = st
                        break
                if not found:
                    return None
                current = found
            return current

        def _count_leaves_and_papers(node: Optional[Dict[str, Any]]) -> Tuple[int, List[str]]:
            """Return (leaf_count, paper_ids) under a node."""
            if not node:
                return 0, []
            subs = node.get("subtopics") or []
            if not subs:
                papers = node.get("papers") or []
                return 1, list(papers)
            leaf_count = 0
            paper_ids: List[str] = []
            for st in subs:
                lc, pids = _count_leaves_and_papers(st)
                leaf_count += lc
                paper_ids.extend(pids)
            return leaf_count, paper_ids

        original_leaf_node = _find_node_by_path(taxonomy, orig_path)
        original_leaf_papers = []
        if original_leaf_node:
            _, original_leaf_papers = _count_leaves_and_papers(original_leaf_node)

        needs_review_warning = ""
        if taxonomy_status == "needs_review":
            needs_review_warning = (
                " Note: the taxonomy is marked as needs_review; structural relationships should be interpreted with caution."
            )

        # Branching
        if sibling_candidate_count == 0 and sibling_subtopic_count > 0:
            # subtopic-level (no per-paper comparisons)
            note = (
                "No sibling papers were found in the same taxonomy leaf. "
                "A taxonomy-subtopic-level comparison will be produced instead."
                + needs_review_warning
            )

            # Build sibling_subtopics payload with representative papers (top2 by rank per subtopic)
            sibling_subtopics_payload: List[Dict[str, Any]] = []
            for sub_name in sibling_subtopic_names:
                sub_path = parent_path + [sub_name]
                sub_node = _find_node_by_path(taxonomy, sub_path)
                leaf_count, paper_ids_under_sub = _count_leaves_and_papers(sub_node)
                papers: List[Dict[str, Any]] = []
                seen_pids = set()
                for pid in paper_ids_under_sub:
                    if pid in seen_pids:
                        continue
                    seen_pids.add(pid)
                    citation_info = self.citation_manager.get_citation_info(pid)
                    title = citation_info.get("title", "") if citation_info else ""
                    abstract = citation_info.get("abstract", "") if citation_info else ""
                    if not (title and abstract):
                        continue  # skip incomplete entries
                    papers.append(
                        {
                            "id": pid,
                            "title": title,
                            "abstract": abstract,
                        }
                    )
                sibling_subtopics_payload.append(
                    {
                        "subtopic_name": sub_name,
                        "scope_note": (sub_node or {}).get("scope_note", ""),
                        "exclude_note": (sub_node or {}).get("exclude_note", ""),
                        "leaf_count": leaf_count,
                        "paper_count": len(paper_ids_under_sub),
                        "papers": papers,
                    }
                )

            subtopic_payload = {
                "taxonomy_path": orig_path,
                "original_leaf": {
                    "leaf_name": leaf_name,
                    "scope_note": (original_leaf_node or {}).get("scope_note", ""),
                    "exclude_note": (original_leaf_node or {}).get("exclude_note", ""),
                    "paper_ids": original_leaf_papers or [taxonomy_original_id],
                },
                "sibling_subtopics": sibling_subtopics_payload,
                "llm_summary_en": None,  # Filled by Phase3 LLMAnalyzer subtopic-level step
                "generated_at": datetime.now().isoformat(),
            }

            try:
                llm_subtopic_summary = self.llm_analyzer.analyze_core_task_subtopics(
                    core_task_text=core_task_text,
                    original_leaf=subtopic_payload["original_leaf"],
                    sibling_subtopics=sibling_subtopics_payload,
                    original_abstract=original_abstract,
                )
                if llm_subtopic_summary:
                    subtopic_payload["llm_summary_en"] = llm_subtopic_summary
            except Exception as e:
                self.logger.warning(f"Phase3: subtopic-level LLM analysis failed: {e}")

            subtopic_file = comparisons_dir / "subtopic_comparison.json"
            with open(subtopic_file, "w", encoding="utf-8") as f:
                json.dump(subtopic_payload, f, ensure_ascii=False, indent=2)

            return _write_empty_outputs(
                position_type="no_siblings_but_subtopic_siblings",
                note_en=note,
                skip_reason="no_siblings_subtopic_level_instead",
                taxonomy_path=orig_path,
                sibling_candidate_count=0,
                sibling_subtopic_count=sibling_subtopic_count,
                used_subtopic_level=True,
                subtopic_comparison_file="subtopic_comparison.json",
            )

        if sibling_candidate_count == 0 and sibling_subtopic_count == 0:
            note = (
                "No sibling papers and no sibling subtopics were found under the same parent taxonomy node; "
                "the paper appears structurally isolated in the taxonomy."
                + needs_review_warning
            )
            return _write_empty_outputs(
                position_type="no_siblings_no_subtopic_siblings",
                note_en=note,
                skip_reason="structurally_isolated",
                taxonomy_path=orig_path,
                sibling_candidate_count=0,
                sibling_subtopic_count=0,
                used_subtopic_level=False,
            )

        # sibling_candidate_count > 0 -> run sibling paper-level comparisons only
        note = (
            "Sibling papers exist in the same taxonomy leaf; core-task comparisons are performed at sibling level."
            + needs_review_warning
        )
        
        # Aggregation buckets
        completed_comparisons: List[Dict[str, Any]] = []
        errored_candidate_ids: List[str] = []
        completed_with_fallback = 0

        # Only keep sibling candidates
        sibling_set = set(sibling_candidate_ids)
        sibling_candidates: List[Tuple[RetrievedPaper, int]] = []
        for c in candidates:
            cid = getattr(c, "canonical_id", None) or c.paper_id
            if cid in sibling_set:
                orig_rank = candidate_meta.get(cid, {}).get("rank")
                if not isinstance(orig_rank, int):
                    orig_rank = len(sibling_candidates) + 1
                sibling_candidates.append((c, orig_rank))
        sibling_candidates.sort(key=lambda x: x[1])

        # Load original paper full text ONLY when needed (sibling comparisons)
        from paper_novelty_pipeline.models import PaperInput
        original_paper_obj = PaperInput(
            paper_id=original_paper_id,
            original_pdf_url=paper_data.get("url") or paper_data.get("original_pdf_url") or "",
            title=original_title,
            abstract=original_abstract,
        )
        original_fulltext = self.pdf_handler.get_original_full_text(original_paper_obj, canonical_id=original_canonical_id)
        if original_fulltext:
            self.logger.info(
                f"Phase3: Loaded {len(original_fulltext)} chars of original paper fulltext"
            )
        else:
            self.logger.warning(
                "Phase3: Failed to load original paper fulltext, will use abstract only"
            )

        # ------------------------------------------------------------------ #
        # Per-candidate processing
        # ------------------------------------------------------------------ #
        def process_single_candidate(candidate: RetrievedPaper, orig_rank: int) -> Optional[Dict[str, Any]]:
            """Process a single core-task candidate.

            Returns:
                - Successful comparison dict (includes 'processing_status': 'processed_sibling'), or
                - A small marker dict with keys:
                    { '_status': 'skipped_non_sibling' | 'errored', 'canonical_id': ... }, or
                - None if an unexpected error occurred.
            """
            try:
                cid = getattr(candidate, "canonical_id", None) or candidate.paper_id

                # Resume mode: reuse existing file if present
                if resume:
                    safe_title = sanitize_id(candidate.title or f"candidate_{orig_rank}")
                    existing_file = comparisons_dir / f"{orig_rank:02d}_{safe_title}.json"
                    if existing_file.exists():
                        self.logger.info(f"Phase3: skipping already processed core-task candidate {cid}")
                        with open(existing_file, "r", encoding="utf-8") as f:
                            return json.load(f)

                # Taxonomy-based context & relationship
                taxonomy_context: Optional[Dict[str, Any]] = None
                relationship = "sibling"
                try:
                    taxonomy_context = self.llm_analyzer._extract_taxonomy_context(
                        taxonomy=taxonomy,
                        original_paper_id=taxonomy_original_id,
                        candidate_paper_id=cid,
                        mapping=taxonomy_mapping,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Phase3: failed to extract taxonomy context for {cid}: {e}"
                    )
                    taxonomy_context = None

                # Only siblings reach here - proceed with FULLTEXT comparison
                self.logger.info(f"Phase3: Processing sibling paper {cid} with FULLTEXT comparison")
                
                # Step 1: Download PDF and extract full text for the candidate
                cand_text, cand_url, comparison_mode, skip_reason = self._prepare_candidate_text(candidate)
                
                if skip_reason and not cand_text:
                    self.logger.warning(
                        f"Failed to get fulltext for sibling {cid}: {skip_reason}. "
                        "Falling back to abstract-only comparison."
                    )
                    # Use abstract as fallback
                    cand_text = getattr(candidate, "abstract", "") or ""
                    comparison_mode = "abstract_fallback"
                else:
                    comparison_mode = "fulltext"
                    self.logger.info(f"Successfully extracted {len(cand_text)} chars of fulltext for {cid}")
                
                # Step 2: LLM-based core task distinction analysis (with fulltext if available)
                analysis_result = self.llm_analyzer.analyze_core_task_distinction(
                    core_task_text=core_task_text,
                    original_title=original_title,
                    original_abstract=original_abstract,
                    original_paper_id=taxonomy_original_id,
                    candidate_title=candidate.title or "",
                    candidate_abstract=getattr(candidate, "abstract", "") or "",
                    candidate_paper_id=cid,
                    candidate_fulltext=cand_text,  # Pass candidate fulltext to LLM
                    original_fulltext=original_fulltext,  # Pass original fulltext to LLM
                    taxonomy_context=taxonomy_context,
                    relationship=relationship,
                )

                if not analysis_result:
                    self.logger.warning(
                        f"Phase3: failed to generate core task analysis for candidate {cid}"
                    )
                    return {
                        "_status": "errored",
                        "canonical_id": cid,
                        "reason": "llm_analysis_failed",
                    }

                # Build comparison result with simplified structure
                comparison_result: Dict[str, Any] = {
                    "rank": orig_rank,
                    "canonical_id": cid,  # Use canonical_id as the primary identifier
                    "candidate_paper_title": candidate.title,
                    "candidate_paper_authors": getattr(candidate, "authors", []) or [],
                    "candidate_paper_url": cand_url or getattr(candidate, "source_url", None) or "",
                    "candidate_paper_abstract": getattr(candidate, "abstract", None) or "",
                    "candidate_paper_venue": getattr(candidate, "venue", None) or "",
                    "candidate_paper_year": getattr(candidate, "year", None) or 0,
                    "relevance_score": getattr(candidate, "relevance_score", 0.0),
                    "original_paper_title": original_title,
                    "original_paper_abstract": original_abstract,
                    "comparison_mode": comparison_mode,  # "fulltext" or "abstract_fallback"
                    "has_taxonomy_context": taxonomy_context is not None,
                    "taxonomy_relationship": relationship,
                    "processed_at": datetime.now().isoformat(),
                    "processing_status": "processed_sibling",
                }
                
                # Add analysis-specific fields (simplified)
                is_duplicate = analysis_result.get("is_duplicate_variant", False)
                comparison_result.update({
                    "is_duplicate_variant": is_duplicate,
                    "brief_comparison": analysis_result.get("brief_comparison", ""),
                })
                
                # Attach textual similarity segments if provided by analyzer
                segments = analysis_result.get("textual_similarity_segments") or []
                normalized_segments: List[Dict[str, Any]] = []
                for seg in segments:
                    if isinstance(seg, dict):
                        normalized_segments.append(seg)
                    else:
                        try:
                            normalized_segments.append(dataclasses.asdict(seg))
                        except Exception:
                            continue
                comparison_result["textual_similarity_segments"] = normalized_segments
                
                # Save individual file
                safe_title = sanitize_id(candidate.title or f"candidate_{orig_rank}")
                individual_file = comparisons_dir / f"{orig_rank:02d}_{safe_title}.json"
                with open(individual_file, "w", encoding="utf-8") as f:
                    json.dump(comparison_result, f, ensure_ascii=False, indent=2)

                self.logger.info(
                    f"Phase3: processed core-task candidate {orig_rank}/{len(sibling_candidates)}: {cid}"
                )
                return comparison_result

            except Exception as e:
                cid = getattr(candidate, "canonical_id", None) or getattr(candidate, "paper_id", "unknown")
                self.logger.error(
                    f"Phase3: error processing core-task candidate {cid}: {e}",
                    exc_info=True,
                )
                return {
                    "_status": "errored",
                    "canonical_id": cid,
                    "reason": "exception",
                }

        # ------------------------------------------------------------------ #
        # Concurrency over candidates
        # ------------------------------------------------------------------ #
        max_workers = max(1, int(PHASE3_CONCURRENCY) if PHASE3_CONCURRENCY else 1)
        from concurrent.futures import ThreadPoolExecutor as _TPE

        with _TPE(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_candidate, candidate, orig_rank): (candidate, orig_rank)
                for candidate, orig_rank in sibling_candidates
            }

            for future, (candidate, orig_rank) in futures.items():
                cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
                try:
                    result = future.result(timeout=300)  # 5 minutes timeout per candidate
                    if isinstance(result, dict):
                        status = result.get("processing_status") or result.get("_status")
                        if status == "processed_sibling" or (
                            not status and "rank" in result
                        ):
                            # Successful sibling comparison
                            completed_comparisons.append(result)
                        elif status in ("errored", "exception"):
                            errored_candidate_ids.append(
                                result.get("canonical_id") or cid
                            )
                        else:
                            # Unknown marker: treat as errored
                            errored_candidate_ids.append(cid)
                    else:
                        # None or unexpected type: treat as errored
                        errored_candidate_ids.append(cid)
                except Exception as e:
                    self.logger.error(
                        f"Phase3: future failed for core-task candidate {cid}: {e}",
                        exc_info=True,
                    )
                    errored_candidate_ids.append(cid)

        # ------------------------------------------------------------------ #
        # Aggregate + write merged files
        # ------------------------------------------------------------------ #
        # Sort successful comparisons by rank
        completed_comparisons.sort(key=lambda x: x.get("rank", 0))

        # Count fallbacks
        # Count any non-fulltext path (explicit abstract_fallback or plain abstract) as fallback
        completed_with_fallback = len(
            [
                c
                for c in completed_comparisons
                if c.get("comparison_mode") in ("abstract_fallback", "abstract")
            ]
        )

        merged_comparisons = {
            "comparisons": completed_comparisons,
            "metadata": {
                "total_comparisons": len(completed_comparisons),
                "generated_at": datetime.now().isoformat(),
            },
        }
        merged_file = comparisons_dir / "core_task_comparisons.json"
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(merged_comparisons, f, ensure_ascii=False, indent=2)

        # Summary (semantic-correct stats; no "successful/failed")
        eligible_candidates = sibling_candidate_count
        attempted = len(sibling_candidates)  # siblings only (including resume loads)
        completed = len(completed_comparisons)
        errored = len(errored_candidate_ids)
        skipped_by_design = max(0, total_candidates - eligible_candidates)

        # Similarity statistics (core-task level). We treat any comparison
        # with at least one textual_similarity_segment as having a similarity
        # warning, analogous to the contribution-level pipeline.
        comparisons_with_similarity_warnings = len(
            [c for c in completed_comparisons if c.get("textual_similarity_segments")]
        )
        total_similarity_segments = sum(
            len(c.get("textual_similarity_segments", [])) for c in completed_comparisons
        )

        summary = {
            "statistics": {
                "total_candidates": total_candidates,
                "eligible_candidates": eligible_candidates,
                "attempted": attempted,
                "completed": completed,
                "completed_with_fallback": completed_with_fallback,
                "errored": errored,
                "skipped_by_design": skipped_by_design,
                "skip_reasons": {"non_sibling": skipped_by_design} if skipped_by_design else {},
                "taxonomy_status": taxonomy_status,
                "sibling_candidate_count": sibling_candidate_count,
                "sibling_subtopic_count": sibling_subtopic_count,
                "used_subtopic_level": False,
                "comparison_mode": "fulltext_or_abstract_fallback",
                "comparisons_with_similarity_warnings": comparisons_with_similarity_warnings,
                "total_similarity_segments": total_similarity_segments,
            },
            "structural_position": {
                "taxonomy_path": orig_path,
                "position_type": "has_siblings",
                "sibling_candidate_count": sibling_candidate_count,
                "sibling_subtopic_count": sibling_subtopic_count,
                "note_en": note,
            },
            "generated_at": datetime.now().isoformat(),
        }
        summary_file = comparisons_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self.logger.info(
            "Phase3: core task comparison completed: "
            f"completed={completed} errored={errored} skipped_by_design={skipped_by_design}"
        )

        return summary
    
    def compare_papers(
        self,
        original_paper: PaperInput,
        extracted_content: ExtractedContent,
        search_results: List[SearchResult],
        original_canonical_id: Optional[str] = None,
    ) -> List[ComparisonResult]:
        """
        Run novelty comparison between the original paper and top-k candidates.
        
        NOTE: This method supports both the new ExtractedContent structure (with 
        contributions field) and the legacy structure (with slots field) for backward 
        compatibility. The new structure is preferred.
        
        Behavior:
        - Flatten all SearchResult groups and sort candidates by relevance_score.
        - Select top_k candidates (if top_k is None or <=0, compare all candidates).
        - Group contributions by candidate to enable batch comparison.
        - Fetch full text (original + candidate) via PDFHandler.
        - For each candidate, run a batch contribution-level comparison workflow.
        - Verify evidence pairs by checking if quotes exist in the full texts (fuzzy match).
        
        Args:
            original_paper: Original paper metadata
            extracted_content: Extracted content with contributions
            search_results: List of search results from Phase2
            original_canonical_id: Optional canonical_id from Phase2 for caching
            
        Returns:
            List of ComparisonResult objects
        """
        # Use provided canonical_id or fallback to paper_id
        orig_id = original_canonical_id or original_paper.paper_id
        self.logger.info(f"Phase3: starting novelty comparison for {orig_id}")
        
        try:
            # Fetch full candidate list
            all_candidate_pairs = self._select_top_candidates(search_results, None)
            if not all_candidate_pairs:
                self.logger.warning("Phase3: no candidates available to compare")
                return []
            
            # Prepare original text once (with caching)
            orig_text = self._get_original_text(original_paper, canonical_id=original_canonical_id)
            
            # Normalize original paper title for comparison
            original_title_norm = normalize_title(getattr(original_paper, "title", None))
            
            # Group contributions by candidate
            candidate_to_contributions: Dict[str, List[Tuple[ASPECT, str, str]]] = defaultdict(list)
            candidate_map: Dict[str, RetrievedPaper] = {}
            
            for candidate, aspect, contribution_name, source_query in all_candidate_pairs:
                # Filter out candidates with identical title to original paper
                cand_title_norm = normalize_title(getattr(candidate, "title", None))
                if original_title_norm and cand_title_norm and cand_title_norm == original_title_norm:
                    self.logger.info(
                        f"Phase3: skipping candidate {candidate.paper_id} because "
                        f"its title '{candidate.title}' is identical to the original paper title."
                    )
                    continue
                
                # Use canonical_id as stable key when available
                candidate_id = getattr(candidate, "canonical_id", None) or candidate.paper_id
                candidate_map[candidate_id] = candidate
                candidate_to_contributions[candidate_id].append((aspect, contribution_name, source_query))
            
            # Sort candidates by highest relevance score
            unique_candidates = []
            for candidate_id, contribution_tuples in candidate_to_contributions.items():
                candidate = candidate_map[candidate_id]
                unique_candidates.append((candidate, contribution_tuples))
            
            # Sort by relevance score (descending)
            unique_candidates.sort(key=lambda x: x[0].relevance_score, reverse=True)
            
            results: List[ComparisonResult] = []
            desired_k = self.top_k if (self.top_k is not None and self.top_k > 0) else len(unique_candidates)
            if desired_k <= 0:
                self.logger.warning("Phase3: desired_k computed as <= 0; returning empty result set")
                return results
            
            # Process with concurrency
            configured_workers = (
                PHASE3_CONCURRENCY if (PHASE3_CONCURRENCY and PHASE3_CONCURRENCY > 0) else 1
            )
            max_workers = max(1, min(configured_workers, desired_k, len(unique_candidates)))
            self.logger.info(
                f"Phase3: processing up to {desired_k} candidates with concurrency={max_workers} "
                f"(configured={configured_workers})"
            )
            
            results = self._process_candidates_concurrent(
                unique_candidates[:desired_k],
                original_paper,
                extracted_content,
                orig_text,
                max_workers,
            )
            
            return results[:desired_k]
            
        except Exception as e:
            self.logger.error(f"Phase3 failed: {e}", exc_info=True)
            return []
    
    def _process_candidates_concurrent(
        self,
        unique_candidates: List[Tuple[RetrievedPaper, List[Tuple[ASPECT, str, str]]]],
        original_paper: PaperInput,
        extracted_content: ExtractedContent,
        orig_text: Optional[str],
        max_workers: int,
    ) -> List[ComparisonResult]:
        """Process candidates concurrently using ThreadPoolExecutor."""
        results: List[ComparisonResult] = []
        candidates_iter = iter(unique_candidates)
        desired_k = len(unique_candidates)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            in_flight: Dict[Any, Tuple[RetrievedPaper, List[Tuple[ASPECT, str, str]]]] = {}
            
            def submit_next() -> bool:
                """Submit the next candidate batch comparison task if needed."""
                if len(results) + len(in_flight) >= desired_k:
                    return False
                try:
                    candidate, contribution_tuples = next(candidates_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    self._process_candidate_with_all_contributions,
                    candidate,
                    contribution_tuples,
                    original_paper,
                    extracted_content,
                    orig_text,
                )
                in_flight[future] = (candidate, contribution_tuples)
                return True
            
            # Prime the executor
            while len(in_flight) < min(max_workers, desired_k) and submit_next():
                pass
            
            if not in_flight:
                return results
            
            while in_flight:
                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    candidate, contribution_tuples = in_flight.pop(future, (None, None))
                    if candidate is None:
                        continue
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                    except Exception as e:
                        cid = candidate.paper_id if candidate else "unknown"
                        self.logger.error(
                            f"Phase3: candidate batch comparison failed for {cid}: {e}"
                        )
                
                # Top up tasks if we still need more successful comparisons
                while (
                    len(results) + len(in_flight) < desired_k
                    and len(in_flight) < max_workers
                    and submit_next()
                ):
                    pass
                
                if not in_flight:
                    break
        
        return results
    
    def _process_candidate_with_all_contributions(
        self,
        candidate: RetrievedPaper,
        contribution_tuples: List[Tuple[ASPECT, str, str]],  # (aspect, contribution_name, source_query)
        original_paper: PaperInput,
        extracted_content: ExtractedContent,
        orig_text: Optional[str],
        timeout: Optional[int] = None,
    ) -> Optional[ComparisonResult]:
        """
        Process a single candidate paper, comparing all its associated contributions in batch.
        
        Args:
            candidate: The candidate paper to compare against
            contribution_tuples: List of (aspect, contribution_name, source_query) tuples for this candidate
            original_paper: Original paper metadata
            extracted_content: Extracted content with contributions
            orig_text: Original paper full text (pre-fetched)
            timeout: Optional timeout override
            
        Returns:
            ComparisonResult on success, otherwise None.
        """
        try:
            cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
            self.logger.info(
                f"Phase3: processing candidate {cid} with {len(contribution_tuples)} contributions"
            )
            
            # Get candidate full text once
            cand_text, candidate_paper_url, comparison_mode, skip_reason = self._prepare_candidate_text(candidate)
            
            if skip_reason and not cand_text:
                # For candidates that cannot be compared, generate a placeholder ComparisonResult
                # with error marker so it doesn't "disappear" in Phase3 statistics
                self.logger.warning(
                    f"Phase3: skip candidate due to PDF/abstract issue: {skip_reason} | id={candidate.paper_id}"
                )
                return self._create_skipped_result(
                    original_paper, candidate, candidate_paper_url, comparison_mode, skip_reason
                )
            
            # Run batch comparison
            self.logger.info(
                f"Phase3: Running batch comparison for candidate {cid} "
                f"({len(contribution_tuples)} contributions)"
            )
            
            # Prepare normalized texts for LLM and verification
            orig_text_normalized, cand_text_normalized = self._prepare_texts_for_comparison(
                orig_text or "",
                cand_text or "",
            )
            
            # Get citation reference for this candidate paper
            candidate_citation = None
            if hasattr(candidate, 'canonical_id') and candidate.canonical_id:
                # Try to find citation index for this canonical_id
                citation_data = self.citation_manager.citation_index.get(candidate.canonical_id)
                if citation_data:
                    # Use the pre-generated alias from citation_index (e.g., "AgentGym-RL[1]")
                    # instead of generic "Alias[X]"
                    alias = citation_data.get('alias', '')
                    index = citation_data.get('index')
                    if alias and index is not None:
                        candidate_citation = f"{alias}[{index}]"
                    elif index is not None:
                        # Fallback to generic format if alias is missing
                        candidate_citation = f"Alias[{index}]"
            
            contribution_analyses = self.llm_analyzer.analyze_candidate_contributions(
                candidate=candidate,
                contribution_tuples=contribution_tuples,
                extracted_content=extracted_content,
                orig_full_text=orig_text or "",
                cand_full_text=cand_text or "",
                candidate_citation=candidate_citation,
                timeout=timeout,
            )
            
            if not contribution_analyses:
                # LLM failed after retries, generate failure marker
                self.logger.error(
                    f"Phase3: Batch comparison failed for candidate {candidate.paper_id}"
                )
                return self._create_failed_result(
                    original_paper, candidate, candidate_paper_url, comparison_mode
                )
            
            # Verify evidence pairs using the same normalized texts
            self.logger.info(
                f"Phase3: Verifying evidence pairs for candidate {cid}"
            )
            for contribution_analysis in contribution_analyses:
                self.evidence_verifier.verify_evidence(
                    contribution_analysis,
                    orig_text_normalized,
                    cand_text_normalized,
                )
            
            # NOTE: Textual similarity segments are now computed by run_textual_similarity.py
            # and stored in phase3/textual_similarity_detection/results.json
            similarity_segments = []
            
            # Build ComparisonResult
            orig_url = self._get_original_url(original_paper)
            primary_query = contribution_tuples[0][2] if contribution_tuples else None
            
            result = ComparisonResult(
                original_paper_id=getattr(original_paper, "canonical_id", None) or original_paper.paper_id,
                original_paper_url=orig_url,
                original_paper_title=getattr(original_paper, 'title', '') or '',
                original_paper_abstract=getattr(original_paper, 'abstract', '') or '',
                original_paper_authors=getattr(original_paper, 'authors', None),
                original_paper_venue=getattr(original_paper, 'venue', '') or '',
                original_paper_year=getattr(original_paper, 'year', None),
                original_paper_keywords=getattr(original_paper, 'keywords', None),
                original_paper_primary_area=getattr(original_paper, 'primary_area', '') or '',
                original_paper_openreview_rating_mean=getattr(original_paper, 'openreview_rating_mean', None),
                candidate_paper_title=candidate.title,
                candidate_paper_url=candidate_paper_url,
                retrieved_paper_id=getattr(candidate, "canonical_id", None) or candidate.paper_id,
                candidate_paper_abstract=getattr(candidate, 'abstract', None),
                candidate_paper_authors=getattr(candidate, 'authors', None),
                candidate_paper_venue=getattr(candidate, 'venue', None),
                candidate_paper_year=getattr(candidate, 'year', None),
                comparison_mode=comparison_mode,
                comparison_note=(f"used_{comparison_mode}_batch" if comparison_mode else None),
                query_source=primary_query,
                analyzed_contributions=contribution_analyses,
                textual_similarity_segments=similarity_segments,
            )
            
            self.logger.info(
                f"Phase3: batch comparison done for candidate {candidate.paper_id} "
                f"({len(contribution_analyses)} contributions analyzed)"
            )
            return result
            
        except Exception as ce:
            self.logger.error(
                f"Phase3: candidate batch comparison failed for {candidate.paper_id}: {ce}",
                exc_info=True,
            )
            return None
    
    def _prepare_candidate_text(
        self,
        candidate: RetrievedPaper
    ) -> Tuple[Optional[str], Optional[str], str, Optional[str]]:
        """
        Get candidate text and determine comparison mode.
        
        Returns:
            Tuple of (cand_text, candidate_paper_url, comparison_mode, skip_reason)
        """
        pdf_url = candidate.pdf_url if candidate.pdf_url and candidate.pdf_url.strip() else None
        skip_reason = None
        
        if pdf_url:
            cand_text, cand_source_type, skip_reason = self.pdf_handler.get_candidate_full_text(
                candidate, sanitize_id
            )
            if cand_source_type == "abstract" or not cand_text:
                candidate_paper_url = candidate.source_url
                comparison_mode = "abstract"
                self.logger.warning(
                    f"Phase3: pdf_url present but falling back to abstract for candidate {candidate.paper_id}; "
                    f"pdf_url={pdf_url} | reason={skip_reason}"
                )
            else:
                candidate_paper_url = pdf_url
                comparison_mode = cand_source_type or "fulltext"
        else:
            cand_text = candidate.abstract or "[MISSING ABSTRACT]"
            candidate_paper_url = candidate.source_url
            comparison_mode = "abstract"
        
        return cand_text, candidate_paper_url, comparison_mode, skip_reason
    
    def _prepare_texts_for_comparison(
        self,
        orig_text: str,
        cand_text: str,
    ) -> Tuple[str, str]:
        """
        Prepare normalized texts for LLM comparison and evidence verification.
        
        Applies the same character budget and normalization as LLM analysis.
        """
        # Apply same character-budget as LLM (but texts have already been
        # truncated at references and MAX_CONTEXT_CHARS upstream).
        orig_text_no_refs = orig_text
        cand_text_no_refs = cand_text
        raw_limit = int(LLM_MAX_PROMPT_CHARS) - 50000
        max_context_chars = min(raw_limit, MAX_CONTEXT_CHARS)
        if max_context_chars < PHASE3_MIN_CONTEXT_CHARS:
            max_context_chars = PHASE3_MIN_CONTEXT_CHARS
        
        orig_text_truncated = (
            orig_text_no_refs[:max_context_chars]
            if len(orig_text_no_refs) > max_context_chars
            else orig_text_no_refs
        )
        cand_text_truncated = (
            cand_text_no_refs[:max_context_chars]
            if len(cand_text_no_refs) > max_context_chars
            else cand_text_no_refs
        )
        
        # Normalize using EvidenceVerifier
        orig_text_normalized = self.evidence_verifier.normalize_text(orig_text_truncated)
        cand_text_normalized = self.evidence_verifier.normalize_text(cand_text_truncated)
        
        return orig_text_normalized, cand_text_normalized
    
    def _get_original_text(self, original_paper: PaperInput, canonical_id: Optional[str] = None) -> Optional[str]:
        """Get original paper full text with caching."""
        cache_key = canonical_id or getattr(original_paper, "paper_id", None) or "__ORIG__"
        if cache_key in self._orig_text_cache:
            return self._orig_text_cache.get(cache_key)
        
        self.logger.info("Phase3: preparing original paper full text ...")
        orig_text = self.pdf_handler.get_original_full_text(original_paper, canonical_id=canonical_id)
        self._orig_text_cache[cache_key] = orig_text
        return orig_text
    
    def _get_original_url(self, original_paper: PaperInput) -> str:
        """Get original paper URL with fallbacks."""
        return (
            original_paper.original_pdf_url 
            or (original_paper.paper_id if isinstance(original_paper.paper_id, str) and original_paper.paper_id.startswith("http") else None)
            or original_paper.paper_id
        )
    
    def _create_skipped_result(
        self,
        original_paper: PaperInput,
        candidate: RetrievedPaper,
        candidate_paper_url: Optional[str],
        comparison_mode: str,
        skip_reason: str,
    ) -> ComparisonResult:
        """Create a ComparisonResult for skipped candidates."""
        return ComparisonResult(
            original_paper_id=getattr(original_paper, "canonical_id", None) or original_paper.paper_id,
            original_paper_url=self._get_original_url(original_paper),
            candidate_paper_title=candidate.title,
            candidate_paper_url=candidate_paper_url or candidate.source_url,
            retrieved_paper_id=getattr(candidate, "canonical_id", None) or candidate.paper_id,
            candidate_paper_abstract=getattr(candidate, 'abstract', None),
            candidate_paper_authors=getattr(candidate, 'authors', None),
            candidate_paper_venue=getattr(candidate, 'venue', None),
            candidate_paper_year=getattr(candidate, 'year', None),
            comparison_mode=comparison_mode or "abstract",
            comparison_note=f"skipped_due_to_pdf_issue:{skip_reason}",
            analyzed_contributions=[],
            textual_similarity_segments=[],
        )
    
    def _create_failed_result(
        self,
        original_paper: PaperInput,
        candidate: RetrievedPaper,
        candidate_paper_url: Optional[str],
        comparison_mode: str,
    ) -> ComparisonResult:
        """Create a ComparisonResult for failed comparisons."""
        orig_url = self._get_original_url(original_paper)
        return ComparisonResult(
            original_paper_id=getattr(original_paper, "canonical_id", None) or original_paper.paper_id,
            original_paper_url=orig_url,
            original_paper_title=getattr(original_paper, 'title', '') or '',
            original_paper_abstract=getattr(original_paper, 'abstract', '') or '',
            original_paper_authors=getattr(original_paper, 'authors', None),
            original_paper_venue=getattr(original_paper, 'venue', '') or '',
            original_paper_year=getattr(original_paper, 'year', None),
            original_paper_keywords=getattr(original_paper, 'keywords', None),
            original_paper_primary_area=getattr(original_paper, 'primary_area', '') or '',
            original_paper_openreview_rating_mean=getattr(original_paper, 'openreview_rating_mean', None),
            candidate_paper_title=candidate.title,
            candidate_paper_url=candidate_paper_url,
            retrieved_paper_id=getattr(candidate, "canonical_id", None) or candidate.paper_id,
            candidate_paper_abstract=getattr(candidate, 'abstract', None),
            candidate_paper_authors=getattr(candidate, 'authors', None),
            candidate_paper_venue=getattr(candidate, 'venue', None),
            candidate_paper_year=getattr(candidate, 'year', None),
            comparison_mode=comparison_mode or "abstract",
            comparison_note="llm_batch_failed",
            analyzed_contributions=[],
            textual_similarity_segments=[],
        )
    
    def _select_top_candidates(
        self,
        search_results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[Tuple[RetrievedPaper, ASPECT, str, str]]:
        """
        DEPRECATED: This method extracts aspect/contribution from SearchQuery,
        which is no longer needed since Phase3 only handles contributions now.
        
        Flatten all search groups and pick top-k by relevance_score (desc).
        Returns: List of (candidate, aspect, contribution_name, source_query) tuples.
        """
        pairs: List[Tuple[RetrievedPaper, ASPECT, str, str]] = []
        for grp in search_results or []:
            # Extract aspect and contribution_name from SearchQuery
            aspect = getattr(grp.query, "aspect", "contribution")  # Default to "contribution"
            contribution_name = getattr(grp.query, "contribution_name", "") or ""
            source_query = getattr(grp.query, "query", "") or ""
            
            for p in grp.retrieved_papers:
                pairs.append((p, aspect, contribution_name, source_query))
        
        if not pairs:
            return []
        
        # Sort by relevance_score (descending)
        pairs.sort(key=lambda x: x[0].relevance_score, reverse=True)
        
        if top_k is None or top_k <= 0:
            return pairs
        return pairs[:top_k]
    
    def _load_phase2_results(
        self,
        phase2_dir: Path,
        contribs: List[Dict[str, Any]],
        max_candidates: int,
        contribution_indices: Optional[List[int]],
        extracted_content: ExtractedContent,
    ) -> List[SearchResult]:
        """
        Load Phase2 search results from contribution files using contribution_mapping.json.
        
        This method uses canonical_id-based mapping to ensure each contribution
        loads its correct TopK papers.
        """
        phase2_final = Path(phase2_dir) / "final"
        mapping_file = phase2_final / "contribution_mapping.json"
        
        # Try to load contribution_mapping.json (new canonical ID-based approach)
        if mapping_file.exists():
            try:
                return self._load_from_canonical_mapping(
                    mapping_file, phase2_final, contribs, max_candidates, contribution_indices
                )
            except Exception as e:
                self.logger.warning(
                    f"Phase3: failed to load from contribution_mapping.json: {e}. Falling back to legacy method."
                )
        
        # Legacy fallback: load from file paths (backward compatibility)
        self.logger.info("Phase3: using legacy file-based loading (contribution_mapping.json not found)")
        return self._load_from_file_paths_legacy(
            phase2_dir, contribs, max_candidates, contribution_indices, extracted_content
        )
    
    def _load_from_canonical_mapping(
        self,
        mapping_file: Path,
        phase2_final: Path,
        contribs: List[Dict[str, Any]],
        max_candidates: int,
        contribution_indices: Optional[List[int]],
    ) -> List[SearchResult]:
        """
        Load contribution papers using canonical_id mapping (new approach).
        
        This ensures each contribution loads exactly its TopK papers based on canonical_id.
        """
        # Load mapping
        mapping_data = json.loads(mapping_file.read_text(encoding="utf-8"))
        contributions_map = mapping_data.get("contributions", [])
        
        self.logger.info(
            f"Phase3: loaded contribution_mapping.json with {len(contributions_map)} contributions"
        )
        
        # Build a lookup: contribution_id -> canonical_ids list
        contrib_id_to_canonical_ids: Dict[str, List[str]] = {}
        for entry in contributions_map:
            contrib_id = entry.get("contribution_id", "")
            canonical_ids = entry.get("canonical_ids", [])
            contrib_id_to_canonical_ids[contrib_id] = canonical_ids
        
        # Load all candidate papers from TopK files and build canonical_id -> paper mapping
        canonical_id_to_paper: Dict[str, Dict[str, Any]] = {}
        
        # Load core task candidates
        core_file = phase2_final / "core_task_perfect_top50.json"
        if core_file.exists():
            items = read_json_list(core_file)
            for item in items:
                cid = item.get("canonical_id")
                if cid:
                    canonical_id_to_paper[cid] = item
        
        # Load all contribution TopK files
        for contrib_file in phase2_final.glob("contribution_*_perfect_top*.json"):
            items = read_json_list(contrib_file)
            for item in items:
                cid = item.get("canonical_id")
                if cid:
                    canonical_id_to_paper[cid] = item
        
        self.logger.info(
            f"Phase3: built canonical_id lookup with {len(canonical_id_to_paper)} papers"
        )
        
        # Build SearchResults for each contribution
        search_results: List[SearchResult] = []
        
        for i, contrib in enumerate(contribs):
            contrib_idx = i + 1  # 1-based indexing
            contrib_id = f"contribution_{contrib_idx}"
            
            # Check if this contribution should be processed
            if isinstance(contribution_indices, list) and contribution_indices:
                if contrib_idx not in contribution_indices:
                    continue
            
            # Get canonical_ids for this contribution
            canonical_ids = contrib_id_to_canonical_ids.get(contrib_id, [])
            
            if not canonical_ids:
                self.logger.warning(
                    f"Phase3: no canonical_ids found for {contrib_id} in mapping"
                )
                continue
            
            # Limit to max_candidates if specified
            if isinstance(max_candidates, int) and max_candidates > 0:
                canonical_ids = canonical_ids[:max_candidates]
            
            # Load papers by canonical_id
            rps: List[RetrievedPaper] = []
            for rank, cid in enumerate(canonical_ids, start=1):
                paper_data = canonical_id_to_paper.get(cid)
                if paper_data:
                    rp = make_retrieved_paper(paper_data, fallback_score=rank)
                    if rp:
                        rps.append(rp)
                else:
                    self.logger.warning(
                        f"Phase3: canonical_id {cid} not found in any TopK file"
                    )
            
            if not rps:
                self.logger.warning(
                    f"Phase3: no papers loaded for {contrib_id}"
                )
                continue
            
            # Create SearchResult
            cname = contrib.get("name") or "contribution"
            qtext = contrib.get("prior_work_query") or cname
            sq = SearchQuery(
                query=qtext,
                aspect="contribution",
                contribution_name=cname,
                source_description=f"Phase2 canonical mapping ({len(rps)} papers)"
            )
            search_results.append(SearchResult(query=sq, retrieved_papers=rps, total_results=len(rps)))
            
            self.logger.info(
                f"Phase3: loaded {len(rps)} papers for {contrib_id} using canonical_id mapping"
            )
        
        if not search_results:
            raise ValueError(
                "No search results loaded from contribution_mapping.json. "
                "This likely indicates a mismatch between Phase1 contributions and Phase2 outputs."
            )
        
        return search_results
    
    def _load_from_file_paths_legacy(
        self,
        phase2_dir: Path,
        contribs: List[Dict[str, Any]],
        max_candidates: int,
        contribution_indices: Optional[List[int]],
        extracted_content: ExtractedContent,
    ) -> List[SearchResult]:
        """
        Legacy method: Load Phase2 results from file paths.
        
        This is kept for backward compatibility with older Phase2 outputs
        that don't have contribution_mapping.json.
        """
        idxp = Path(phase2_dir) / "final" / "index.json"
        contrib_files: List[str] = []
        
        if idxp.exists():
            try:
                idx = json.loads(idxp.read_text(encoding="utf-8"))
                cf = idx.get("contribution_files")
                if isinstance(cf, list):
                    contrib_files = [str(x) for x in cf]
            except Exception:
                pass
        
        if not contrib_files:
            alt = Path(phase2_dir) / "final" / "contributions_index_top10.json"
            if alt.exists():
                arr = json.loads(alt.read_text(encoding="utf-8"))
                if isinstance(arr, list):
                    for entry in arr:
                        p = (entry or {}).get("output_file")
                        if isinstance(p, str):
                            contrib_files.append(p)
        
        # Build SearchResults list
        search_results: List[SearchResult] = []
        for fp in contrib_files:
            p = Path(fp)
            if not p.is_absolute():
                # Phase2 postprocess may store paths as:
                #  - project-root relative: "output/.../phase2/final/contribution_*.json"
                #  - phase2_dir relative:   "final/contribution_*.json"
                # We resolve in a tolerant order:
                # 1) project-root relative (preferred for modern pipeline)
                # 2) phase2_dir relative (legacy)
                cand1 = Path(PROJECT_ROOT) / p
                if cand1.exists():
                    p = cand1
                else:
                    p = Path(phase2_dir) / p
            if not p.exists():
                continue
            
            items = read_json_list(p)
            if isinstance(max_candidates, int) and max_candidates > 0:
                items = items[:max_candidates]
            
            rps: List[RetrievedPaper] = []
            for rank, it in enumerate(items, start=1):
                rp = make_retrieved_paper(it, fallback_score=rank)
                if rp:
                    rps.append(rp)
            
            # Determine contribution index
            m = re.search(r"contribution_(\d+)", p.stem)
            if m:
                idx = int(m.group(1))
                if isinstance(contribution_indices, list) and contribution_indices and idx not in contribution_indices:
                    continue
                cname = contribs[idx - 1].get("name") if idx <= len(contribs) else None
            else:
                cname = None
            
            if not cname:
                # Fallback: get from extracted_content if available
                if hasattr(extracted_content, 'contributions') and extracted_content.contributions:
                    cname = extracted_content.contributions[0].name
                elif hasattr(extracted_content, 'slots') and extracted_content.slots and extracted_content.slots[0].contributions:
                    cname = extracted_content.slots[0].contributions[0].name
                else:
                    cname = "contribution"
            
            qtext = next((c.get("prior_work_query") for c in contribs if (c.get("name") or "contribution") == cname), cname)
            sq = SearchQuery(query=qtext, aspect="contribution", contribution_name=cname, source_description="Phase2 final TopK (legacy)")
            search_results.append(SearchResult(query=sq, retrieved_papers=rps, total_results=len(rps)))
        
        # REMOVED: Incorrect fallback to core_task_perfect_top50.json
        # This was causing all contributions to compare the same papers
        if not search_results:
            raise ValueError(
                "No contribution files found in Phase2 outputs. "
                "Please ensure Phase2 ran successfully and generated contribution TopK files."
            )
        
        return search_results
    
    def _load_phase1_contributions(self, base_dir: str) -> List[Dict[str, Any]]:
        """
        Load all contributions defined in Phase1 extraction.
        
        Supports both new format (direct "contributions" field) and old format ("slots" field).
        
        Returns:
            List of dicts with keys: aspect, name, description, query
        """
        phase1_path = os.path.join(base_dir, "phase1", PHASE1_EXTRACTED_JSON)
        if not os.path.exists(phase1_path):
            self.logger.warning(f"Phase1 file not found: {phase1_path}")
            return []
        
        try:
            with open(phase1_path, "r", encoding="utf-8") as f:
                phase1_data = json.load(f)
            
            all_contributions = []
            
            # New format: direct contributions field
            if "contributions" in phase1_data:
                contributions_list = phase1_data.get("contributions", [])
                if isinstance(contributions_list, list):
                    for contrib in contributions_list:
                        if isinstance(contrib, dict):
                            all_contributions.append({
                                "aspect": "contribution",  # Currently only handling contribution
                                "name": contrib.get("name"),
                                "description": contrib.get("description") or contrib.get("author_claim_text", ""),
                                "query": contrib.get("prior_work_query", "")
                            })
            
            # Legacy format: slots field
            elif "slots" in phase1_data:
                slots_list = phase1_data.get("slots", [])
                if isinstance(slots_list, list):
                    for slot_group in slots_list:
                        if isinstance(slot_group, dict):
                            aspect = slot_group.get("aspect", "contribution")
                            contributions = slot_group.get("contributions", [])
                            for contrib in contributions:
                                if isinstance(contrib, dict):
                                    all_contributions.append({
                                        "aspect": aspect,
                                        "name": contrib.get("name"),
                                        "description": contrib.get("description", ""),
                                        "query": contrib.get("prior_work_query", "")
                                    })
            
            return all_contributions
            
        except Exception as e:
            self.logger.warning(f"Failed to load Phase1 contributions: {e}")
            return []
    
    def _write_contribution_projection(
        self,
        out_dir: Path,
        contributions: List[Dict[str, Any]],
        results: List[ComparisonResult],
    ) -> None:
        """Write contribution-centric projection summary with 1-based indexing and refutation-based format."""
        base = Path(out_dir).resolve() / "phase3" / "contribution_analysis"  # Use absolute path to avoid creating dirs in wrong location
        os.makedirs(base, exist_ok=True)
        
        # Map contribution name -> index + meta (1-based indexing)
        name_to_idx: Dict[str, int] = {}
        for i, c in enumerate(contributions):
            contrib_num = i + 1  # 1-based indexing
            name_to_idx[c.get("name") or "contribution"] = contrib_num
            cdir = base / f"contribution_{contrib_num}"  # contribution_1, contribution_2, etc.
            os.makedirs(cdir, exist_ok=True)
            meta = {
                "contribution_index": contrib_num,  # 1-based
                "contribution_name": c.get("name") or "",
                "author_claim_text": c.get("author_claim_text") or "",
                "description": c.get("description") or "",
                "source_hint": c.get("source_hint") or "",
            }
            with open(cdir / "contribution_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        
        # Aggregate per contribution
        per_counts: Dict[int, Dict[str, int]] = {}
        for res in results:
            for contrib_analysis in res.analyzed_contributions:
                if contrib_analysis.aspect != "contribution":
                    continue
                cname = contrib_analysis.contribution_name
                contrib_num = name_to_idx.get(cname)
                if contrib_num is None:
                    continue
                
                cdir = base / f"contribution_{contrib_num}"  # 1-based directory
                
                # Build item based on refutation_status
                item = {
                    "canonical_id": getattr(res, 'canonical_id', None) or res.retrieved_paper_id,
                    "title": res.candidate_paper_title,
                    "url": res.candidate_paper_url,
                    "refutation_status": contrib_analysis.refutation_status or "unclear",
                    "comparison_mode": res.comparison_mode,
                    "textual_similarity_segments": [
                        dataclasses.asdict(seg) if not isinstance(seg, dict) else seg 
                        for seg in (res.textual_similarity_segments or [])
                    ],
                }
                self.logger.info(f"DEBUG: writing item for {res.retrieved_paper_id} with {len(item.get('textual_similarity_segments', []))} similarity segments")
                
                # Add fields based on refutation_status
                if contrib_analysis.refutation_status == "can_refute":
                    # Can refute: add refutation evidence
                    if contrib_analysis.refutation_evidence:
                        ref_ev = contrib_analysis.refutation_evidence
                        item["refutation_evidence"] = {
                            "summary": ref_ev.get("summary", ""),
                            "evidence_pairs": [
                                {
                                    "original_quote": ep.original_quote,
                                    "candidate_quote": ep.candidate_quote,
                                    "rationale": ep.rationale,
                                    "original_location": {
                                        "paragraph_label": ep.original_location.paragraph_label,
                                        "found": ep.original_location.found,
                                        "match_score": ep.original_location.match_score,
                                    },
                                    "candidate_location": {
                                        "paragraph_label": ep.candidate_location.paragraph_label,
                                        "found": ep.candidate_location.found,
                                        "match_score": ep.candidate_location.match_score,
                                    },
                                }
                                for ep in (ref_ev.get("evidence_pairs") or [])
                            ],
                        }
                else:
                    # Cannot refute or unclear: add brief note
                    item["brief_note"] = contrib_analysis.brief_note or ""
                
                with open(cdir / f"paper_{sanitize_id(str(res.retrieved_paper_id))}.json", "w", encoding="utf-8") as f:
                    json.dump(item, f, ensure_ascii=False, indent=2)
                
                # Count based on whether there's evidence
                counts = per_counts.setdefault(contrib_num, {"with_evidence": 0, "no_evidence": 0})
                has_evidence = (
                    contrib_analysis.refutation_status == "can_refute" and contrib_analysis.refutation_evidence
                )
                if has_evidence:
                    counts["with_evidence"] += 1
                else:
                    counts["no_evidence"] += 1
        
        # Write summary files (1-based indexing)
        for i, c in enumerate(contributions):
            contrib_num = i + 1  # 1-based
            cdir = base / f"contribution_{contrib_num}"
            counts = per_counts.get(contrib_num, {"with_evidence": 0, "no_evidence": 0})
            summ = {
                "contribution_index": contrib_num,  # 1-based
                "contribution_name": c.get("name") or "",
                "counts_by_simple_evidence": counts,
            }
            with open(cdir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summ, f, ensure_ascii=False, indent=2)

    def _extend_citation_index_for_contribution_analysis(
        self,
        search_results: List[SearchResult],
    ) -> None:
        """
        Extend citation_index with candidate papers used in Contribution Analysis.
        """
        for group in search_results or []:
            for candidate in group.retrieved_papers:
                # Prefer canonical_id as the citation key when available
                cid = getattr(candidate, "canonical_id", None) or candidate.paper_id
                if cid in self.citation_manager.citation_index:
                    continue
                self.citation_manager.add_paper(
                    paper_id=cid,
                    title=candidate.title or "",
                    year=candidate.year,
                    url=candidate.source_url,
                    authors=candidate.authors,
                    venue=candidate.venue,
                    doi=candidate.doi,
                    arxiv_id=candidate.arxiv_id,
                )
