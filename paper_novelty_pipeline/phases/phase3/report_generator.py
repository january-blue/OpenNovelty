"""
Phase 3: Report Generator

Handles generation and saving of Phase3 reports including statistics,
aggregated results, and complete merged reports.
"""

import os
import json
import logging
import dataclasses
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any, Callable

from paper_novelty_pipeline.models import (
    PaperInput,
    SearchResult,
    ComparisonResult,
    RetrievedPaper,
)
from paper_novelty_pipeline.utils.paths import PAPER_JSON, PHASE1_EXTRACTED_JSON

from paper_novelty_pipeline.phases.phase3.utils import (
    read_json_list,
    get_comparison_filename,
    extract_authors,
)
from paper_novelty_pipeline.phases.phase3.citation_manager import CitationManager
from paper_novelty_pipeline.services.llm_client import create_llm_client
from paper_novelty_pipeline.utils.report_artifacts import (
    phase4_lightweight_md_filename_from_phase3_report,
    phase4_lightweight_pdf_filename_from_phase3_report,
)


# ============================================================================
# ReportGenerator Class
# ============================================================================

class ReportGenerator:
    """
    Generates and saves Phase3 reports including statistics and aggregated results.
    
    This class handles:
    - Individual comparison report saving
    - Statistics generation (by aspect, by contribution, overall)
    - Main Phase3 report generation with aggregated results
    - Complete report merging (survey + contribution analysis + core task comparisons)
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        output_base_dir: Optional[str] = None,
        load_phase1_contributions: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
        citation_manager: Optional[CitationManager] = None,
    ):
        """
        Initialize ReportGenerator.
        
        Args:
            logger: Logger instance
            output_base_dir: Base directory for output files
            load_phase1_contributions: Optional function to load Phase1 contributions.
                                      If None, will use default implementation.
            citation_manager: Citation manager instance. If None, one will be created.
        """
        self.logger = logger
        self.output_base_dir = output_base_dir or os.getcwd()
        self._load_phase1_contributions = load_phase1_contributions or self._default_load_phase1_contributions
        
        # Initialize citation manager
        base_dir_path = Path(self.output_base_dir)
        self.citation_manager = citation_manager or CitationManager(
            logger=self.logger,
            base_dir=base_dir_path
        )
        # Try to load citation_index from survey_report
        self.citation_manager.load_from_survey_report()
    
    def _default_load_phase1_contributions(self, base_dir: str) -> List[Dict[str, Any]]:
        """
        Default implementation to load Phase1 contributions.
        
        Supports both new format (direct "contributions" field) and old format ("slots" field).
        
        Args:
            base_dir: Base directory containing phase1 subdirectory
            
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
            
            # New format: direct "contributions" field
            if "contributions" in phase1_data:
                contributions_list = phase1_data.get("contributions", [])
                if isinstance(contributions_list, list):
                    for contrib in contributions_list:
                        if isinstance(contrib, dict):
                            all_contributions.append({
                                "aspect": "contribution",  # Currently only handles contribution
                                "name": contrib.get("name"),
                                "description": contrib.get("description") or contrib.get("author_claim_text", ""),
                                "query": contrib.get("prior_work_query", "")
                            })
            
            # Legacy format: "slots" field
            elif "slots" in phase1_data:
                for group in phase1_data.get("slots", []):
                    # Support both old format (subslots) and new format (contributions)
                    contributions_data = group.get("contributions", group.get("subslots", []))
                    for contribution in contributions_data:
                        if isinstance(contribution, dict):
                            all_contributions.append({
                                "aspect": contribution.get("aspect", "contribution"),
                                "name": contribution.get("name"),
                                "description": contribution.get("description", ""),
                                "query": contribution.get("prior_work_query", "")
                            })
            
            self.logger.info(f"Loaded {len(all_contributions)} contributions from Phase1")
            return all_contributions
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(f"Failed to load Phase1 contributions: {e}")
            return []
    
    # ========================================================================
    # Individual Report Saving
    # ========================================================================
    
    def save_comparison_report(
        self,
        result: ComparisonResult,
        original_paper: PaperInput,
        candidate: RetrievedPaper,
    ) -> None:
        """
        Save individual comparison result to JSON files with descriptive naming.
        
        Saves to:
        - contribution_analysis/comparisons/{descriptive_filename}.json (individual report)
        - contribution_analysis/phase3_report.jsonl (aggregated JSONL)
        
        Args:
            result: ComparisonResult to save
            original_paper: Original paper input
            candidate: Candidate paper that was compared
        """
        try:
            base_dir = Path(self.output_base_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
            
            # Save to contribution_analysis/comparisons/ subdirectory
            comparisons_dir = base_dir / "phase3" / "contribution_analysis" / "comparisons"
            comparisons_dir.mkdir(parents=True, exist_ok=True)
            
            # Get current index
            existing_files = [f for f in os.listdir(comparisons_dir) if f.endswith('.json')]
            index = len(existing_files) + 1
            
            # Get contribution info from first analysis
            if result.analyzed_contributions:
                first_contribution = result.analyzed_contributions[0]
                aspect = first_contribution.aspect
                contribution_name = first_contribution.contribution_name
            else:
                aspect = "contribution"
                contribution_name = "unknown"
            
            # Generate descriptive filename using utils
            filename = get_comparison_filename(
                index=index,
                aspect=aspect,
                contribution_name=contribution_name,
                candidate_id=candidate.paper_id
            )
            
            report_json_path = comparisons_dir / filename
            
            # Save individual comparison
            with open(report_json_path, "w", encoding="utf-8") as fj:
                json.dump(dataclasses.asdict(result), fj, ensure_ascii=False, indent=2)
            
            # Append to aggregated JSONL
            try:
                aggregate_path = base_dir / "phase3" / "contribution_analysis" / "phase3_report.jsonl"
                aggregate_path.parent.mkdir(parents=True, exist_ok=True)
                with open(aggregate_path, "a", encoding="utf-8") as agg_f:
                    json.dump(dataclasses.asdict(result), agg_f, ensure_ascii=False)
                    agg_f.write("\n")
            except (IOError, OSError) as agg_err:
                self.logger.warning(f"Failed to append to JSONL: {agg_err}")
            
            self.logger.info(f"Wrote Phase3 comparison: {filename}")
        except (IOError, OSError, TypeError) as re_err:
            self.logger.warning(f"Failed to write Phase3 report: {re_err}")
    
    # ========================================================================
    # Statistics Generation
    # ========================================================================
    
    def generate_statistics(
        self,
        search_results: List[SearchResult],
        comparison_results: List[ComparisonResult],
        phase1_contributions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about search and comparison results.
        
        Includes all Phase1-defined contributions (even if not searched in Phase2).
        
        Args:
            search_results: List of search results from Phase2
            comparison_results: List of comparison results from Phase3
            phase1_contributions: List of contributions defined in Phase1
            
        Returns:
            Dictionary with keys: summary, by_aspect, by_contribution, comparison
        """
        # Collect contribution descriptions
        contribution_desc_map = self._collect_contribution_descriptions(
            search_results, comparison_results
        )
        
        # Calculate statistics
        aspect_stats = self._calculate_aspect_stats(search_results)
        contribution_stats = self._calculate_contribution_stats(
            search_results, contribution_desc_map
        )
        comparison_stats = self._calculate_comparison_stats(comparison_results)
        
        # Fill in Phase1 contributions that weren't searched
        contribution_stats = self._fill_missing_contributions(
            contribution_stats, phase1_contributions, contribution_desc_map
        )
        
        # Convert to serializable format
        aspect_summary = {
            aspect: {
                "total_retrieved": data["total_retrieved"],
                "unique_candidates": len(data["unique_candidates"]),
            }
            for aspect, data in aspect_stats.items()
        }
        
        return {
            "summary": {
                "total_aspects": len(aspect_summary),
                "total_search_queries": len(search_results),
                "total_unique_candidates": len(set(
                    paper.paper_id 
                    for sr in search_results 
                    for paper in sr.retrieved_papers
                )),
            },
            "by_aspect": aspect_summary,
            "by_contribution": contribution_stats,
            "comparison": comparison_stats,
        }
    
    def _collect_contribution_descriptions(
        self,
        search_results: List[SearchResult],
        comparison_results: List[ComparisonResult],
    ) -> Dict[Tuple[str, str], str]:
        """Collect contribution descriptions from comparison and search results."""
        contribution_desc_map: Dict[Tuple[str, str], str] = {}
        
        # From comparison_results
        for comp in comparison_results:
            for contribution_analysis in comp.analyzed_contributions:
                key = (contribution_analysis.aspect, contribution_analysis.contribution_name)
                desc = contribution_analysis.contribution_description or ""
                if desc and key not in contribution_desc_map:
                    contribution_desc_map[key] = desc
        
        # From search_results (Phase2 source_description)
        for sr in search_results:
            key = (sr.query.aspect, sr.query.contribution_name)
            if key not in contribution_desc_map:
                desc = sr.query.source_description or ""
                if desc:
                    contribution_desc_map[key] = desc
        
        return contribution_desc_map
    
    def _calculate_aspect_stats(
        self,
        search_results: List[SearchResult],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics grouped by aspect."""
        aspect_stats = defaultdict(lambda: {"total_retrieved": 0, "unique_candidates": set()})
        
        for search_result in search_results:
            aspect = search_result.query.aspect
            num_candidates = len(search_result.retrieved_papers)
            
            aspect_stats[aspect]["total_retrieved"] += num_candidates
            for paper in search_result.retrieved_papers:
                aspect_stats[aspect]["unique_candidates"].add(paper.paper_id)
        
        return dict(aspect_stats)
    
    def _calculate_contribution_stats(
        self,
        search_results: List[SearchResult],
        contribution_desc_map: Dict[Tuple[str, str], str],
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics grouped by contribution."""
        contribution_stats = {}
        
        for search_result in search_results:
            query = search_result.query
            aspect = query.aspect
            contribution_name = query.contribution_name
            num_candidates = len(search_result.retrieved_papers)
            
            key = (aspect, contribution_name)
            contribution_stats[contribution_name] = {
                "aspect": aspect,
                "total_retrieved": num_candidates,
                "unique_candidates": len(set(p.paper_id for p in search_result.retrieved_papers)),
                "query": query.query,
                "contribution_description": contribution_desc_map.get(key, ""),
            }
        
        return contribution_stats
    
    def _calculate_comparison_stats(
        self,
        comparison_results: List[ComparisonResult],
    ) -> Dict[str, Any]:
        """Calculate statistics about comparisons."""
        return {
            "total_comparisons": len(comparison_results),
            "successful_comparisons": len([r for r in comparison_results if r.analyzed_contributions]),
            "failed_comparisons": len([r for r in comparison_results if not r.analyzed_contributions]),
            "refutations_downgraded_by_audit": 0,  # Will be updated by _enforce_verified_refutation_policy
            "comparisons_with_similarity_warnings": len([
                r for r in comparison_results if r.textual_similarity_segments
            ]),
            "total_contribution_analyses": sum(
                len(r.analyzed_contributions) for r in comparison_results
            ),
            "total_similarity_segments": sum(
                len(r.textual_similarity_segments) for r in comparison_results
            ),
            "comparison_modes": {
                "fulltext": len([r for r in comparison_results if r.comparison_mode == "fulltext"]),
                "abstract": len([r for r in comparison_results if r.comparison_mode == "abstract"]),
            }
        }
    
    def _fill_missing_contributions(
        self,
        contribution_stats: Dict[str, Dict[str, Any]],
        phase1_contributions: List[Dict[str, Any]],
        contribution_desc_map: Dict[Tuple[str, str], str],
    ) -> Dict[str, Dict[str, Any]]:
        """Fill in Phase1 contributions that weren't searched in Phase2."""
        for p1_contrib in phase1_contributions:
            contribution_name = p1_contrib["name"]
            if contribution_name not in contribution_stats:
                # Phase2 didn't search this contribution, fill with 0
                key = (p1_contrib["aspect"], contribution_name)
                contribution_stats[contribution_name] = {
                    "aspect": p1_contrib["aspect"],
                    "total_retrieved": 0,
                    "unique_candidates": 0,
                    "query": p1_contrib["query"],
                    "contribution_description": contribution_desc_map.get(
                        key, p1_contrib["description"]
                    )
                }
        
        return contribution_stats
    
    # ========================================================================
    # Main Report Generation
    # ========================================================================
    
    def save_phase3_main_report(
        self,
        original_paper: PaperInput,
        search_results: List[SearchResult],
        comparison_results: List[ComparisonResult],
        phase1_contributions: List[Dict[str, Any]],
    ) -> None:
        """
        Generate and save the main Phase3 report with statistics.
        
        Creates phase3_report.json containing:
        - Original paper metadata
        - All comparison results (split into with/without results)
        - Statistics (per-aspect, per-contribution, overall)
        
        Args:
            original_paper: Original paper input
            search_results: List of search results from Phase2
            comparison_results: List of comparison results from Phase3
            phase1_contributions: List of contributions defined in Phase1
        """
        base_dir = Path(self.output_base_dir).resolve()  # Use absolute path to avoid creating dirs in wrong location
        phase3_dir = base_dir / "phase3"
        contribution_analysis_dir = phase3_dir / "contribution_analysis"
        phase3_dir.mkdir(parents=True, exist_ok=True)
        contribution_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate statistics
        statistics = self.generate_statistics(
            search_results, comparison_results, phase1_contributions
        )
        
        # Aggregate by contribution
        contribution_query_map = self._build_contribution_query_map(search_results)
        contribution_stats_map = self._build_contribution_stats_map(statistics)
        contributions_list = self._aggregate_by_contribution(
            comparison_results, contribution_query_map, contribution_stats_map
        )
        
        # Find contributions with no results
        contributions_no_results = self._find_contributions_no_results(
            phase1_contributions, contributions_list
        )
        
        # Update statistics summary with contribution coverage
        statistics = self._update_statistics_summary(
            statistics, phase1_contributions, contributions_list, contributions_no_results
        )
        
        self.logger.info(
            f"Contributions breakdown: {len(contributions_list)} with results, "
            f"{len(contributions_no_results)} with no results"
        )
        
        # Build complete report
        report = {
            "original_paper": self._extract_original_paper_metadata(original_paper),
            "statistics": statistics,
            "contributions_with_results": contributions_list,
            "contributions_no_results": contributions_no_results,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "1.0",
                "phase3_version": "optimized_batch_with_similarity_detection",
            }
        }
        
        # CRITICAL: Apply the Verified Refutation Policy to the final report 
        # before saving. This ensures that any remaining hallucinations 
        # (after LLMAnalyzer retries) are forcibly downgraded and flagged.
        self._enforce_verified_refutation_policy(report)
        
        # Save main report
        report_path = contribution_analysis_dir / "phase3_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # Generate references markdown file
        # Use canonical_id for citation (backward compatible with retrieved_paper_id)
        cited_paper_ids = [
            getattr(comp, 'canonical_id', None) or getattr(comp, 'retrieved_paper_id', None)
            for comp in comparison_results
            if getattr(comp, 'canonical_id', None) or getattr(comp, 'retrieved_paper_id', None)
        ]
        if cited_paper_ids:
            references_markdown = self.citation_manager.generate_references_markdown(cited_paper_ids)
            references_file = contribution_analysis_dir / "references.md"
            with open(references_file, "w", encoding="utf-8") as f:
                f.write(references_markdown)
        
        self.logger.info(f"Saved Phase3 contribution analysis report to: {report_path}")
        self.logger.info(
            f"Phase3 Summary: {statistics['summary']['total_unique_candidates']} unique candidates, "
            f"{statistics['comparison']['total_comparisons']} comparisons, "
            f"{statistics['comparison']['comparisons_with_similarity_warnings']} with similarity warnings"
        )
    
    def _build_contribution_query_map(
        self,
        search_results: List[SearchResult],
    ) -> Dict[Tuple[str, str], str]:
        """Build mapping from (aspect, contribution_name) to query."""
        contribution_query_map: Dict[Tuple[str, str], str] = {}
        for sr in search_results:
            q = sr.query
            key = (q.aspect, q.contribution_name)
            if key not in contribution_query_map:
                contribution_query_map[key] = q.query
        return contribution_query_map
    
    def _build_contribution_stats_map(
        self,
        statistics: Dict[str, Any],
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Build mapping from (aspect, contribution_name) to stats."""
        by_contribution = statistics.get("by_contribution", {}) or {}
        contribution_stats_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for name, stats in by_contribution.items():
            aspect = stats.get("aspect")
            if not aspect:
                continue
            key = (aspect, name)
            contribution_stats_map[key] = stats
        return contribution_stats_map
    
    def _aggregate_by_contribution(
        self,
        comparison_results: List[ComparisonResult],
        contribution_query_map: Dict[Tuple[str, str], str],
        contribution_stats_map: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Aggregate comparison results by contribution."""
        contribution_groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        
        for comp in comparison_results:
            # Each ComparisonResult may contain multiple contribution analyses
            for contrib_analysis in comp.analyzed_contributions:
                key = (contrib_analysis.aspect, contrib_analysis.contribution_name)
                
                group = contribution_groups.get(key)
                if group is None:
                    stats = contribution_stats_map.get(key, {})
                    desc = contrib_analysis.contribution_description or stats.get("contribution_description", "")
                    query = contribution_query_map.get(key, stats.get("query", ""))
                    
                    group = {
                        "aspect": contrib_analysis.aspect,
                        "contribution_name": contrib_analysis.contribution_name,
                        "contribution_description": desc,
                        "query": query,
                        "stats": {
                            "total_retrieved": stats.get("total_retrieved", 0),
                            "unique_candidates": stats.get("unique_candidates", 0),
                        },
                        "comparisons": [],
                    }
                    contribution_groups[key] = group
                
                # Get citation for candidate (use canonical_id, fallback to retrieved_paper_id)
                paper_id = getattr(comp, 'canonical_id', None) or getattr(comp, 'retrieved_paper_id', None)
                candidate_citation = self.citation_manager.cite(paper_id)
                
                # Add candidate-level comparison summary with new refutation-based structure
                comp_entry = {
                    "candidate_paper_title": comp.candidate_paper_title,
                    "candidate_paper_url": comp.candidate_paper_url,
                    "canonical_id": getattr(comp, 'canonical_id', None) or getattr(comp, 'retrieved_paper_id', None),
                    "candidate_citation": candidate_citation,  # New: unified citation format
                    "comparison_mode": comp.comparison_mode,
                    "comparison_note": comp.comparison_note,
                    "refutation_status": contrib_analysis.refutation_status or "unclear",
                    "verification_warning": getattr(contrib_analysis, 'verification_warning', False),
                    "plagiarism_segments": [
                        dataclasses.asdict(seg) for seg in comp.textual_similarity_segments
                    ],
                }
                
                # Add refutation_evidence if can_refute
                if contrib_analysis.refutation_status == "can_refute" and contrib_analysis.refutation_evidence:
                    ref_ev = contrib_analysis.refutation_evidence
                    comp_entry["refutation_evidence"] = {
                        "summary": ref_ev.get("summary", ""),
                        "evidence_pairs": [
                            dataclasses.asdict(ep) for ep in (ref_ev.get("evidence_pairs") or [])
                        ]
                    }
                # Add brief_note if cannot_refute or unclear
                elif contrib_analysis.refutation_status in ("cannot_refute", "unclear"):
                    if contrib_analysis.brief_note:
                        comp_entry["brief_note"] = contrib_analysis.brief_note
                
                group["comparisons"].append(comp_entry)
        
        # Convert to list and sort for stable output
        contributions_list = list(contribution_groups.values())
        contributions_list.sort(key=lambda g: (g["aspect"], g["contribution_name"]))
        
        return contributions_list
    
    def _find_contributions_no_results(
        self,
        phase1_contributions: List[Dict[str, Any]],
        contributions_list: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Find Phase1 contributions that had no search results."""
        contributions_with_results_set = set(
            (g["aspect"], g["contribution_name"]) for g in contributions_list
        )
        
        contributions_no_results = []
        for p1_contrib in phase1_contributions:
            key = (p1_contrib["aspect"], p1_contrib["name"])
            if key not in contributions_with_results_set:
                contributions_no_results.append({
                    "aspect": p1_contrib["aspect"],
                    "contribution_name": p1_contrib["name"],
                    "contribution_description": p1_contrib["description"],
                    "query": p1_contrib["query"],
                    "search_metadata": {
                        "total_retrieved": 0,
                        "reason": "no_matching_papers",
                        "searched_at": datetime.now().isoformat()
                    }
                })
        
        # Sort by aspect + name
        contributions_no_results.sort(key=lambda x: (x["aspect"], x["contribution_name"]))
        
        return contributions_no_results
    
    def _update_statistics_summary(
        self,
        statistics: Dict[str, Any],
        phase1_contributions: List[Dict[str, Any]],
        contributions_list: List[Dict[str, Any]],
        contributions_no_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update statistics summary with contribution coverage information."""
        summary = statistics.get("summary", {})
        if phase1_contributions:
            summary["total_contributions_defined"] = len(phase1_contributions)
            summary["total_contributions_with_results"] = len(contributions_list)
            summary["total_contributions_no_results"] = len(contributions_no_results)
            summary["coverage_rate"] = round(
                len(contributions_list) / len(phase1_contributions), 3
            )
        
        # Rebuild summary in schema order
        statistics["summary"] = {
            "total_aspects": summary.get("total_aspects"),
            "total_contributions_defined": summary.get("total_contributions_defined"),
            "total_contributions_with_results": summary.get("total_contributions_with_results"),
            "total_contributions_no_results": summary.get("total_contributions_no_results"),
            "coverage_rate": summary.get("coverage_rate"),
            "total_search_queries": summary.get("total_search_queries"),
            "total_unique_candidates": summary.get("total_unique_candidates"),
        }
        
        return statistics
    
    def _extract_original_paper_metadata(self, original_paper: PaperInput) -> Dict[str, Any]:
        """Extract metadata from original paper input."""
        return {
            "canonical_id": getattr(original_paper, 'canonical_id', None) or original_paper.paper_id,
            "title": getattr(original_paper, 'title', ''),
            "abstract": getattr(original_paper, 'abstract', ''),
            "authors": getattr(original_paper, 'authors', None),
            "venue": getattr(original_paper, 'venue', ''),
            "year": getattr(original_paper, 'year', None),
            "doi": getattr(original_paper, 'doi', None),
            "arxiv_id": getattr(original_paper, 'arxiv_id', None),
            "keywords": getattr(original_paper, 'keywords', None),
            "primary_area": getattr(original_paper, 'primary_area', ''),
            "openreview_rating_mean": getattr(original_paper, 'openreview_rating_mean', None),
        }
    
    # ========================================================================
    # Complete Report Generation
    # ========================================================================
    
    def generate_complete_report(
        self,
        out_dir: Path,
        *,
        include_survey: bool = True,
        include_contribution: bool = True,
        include_core_task: bool = True,
    ) -> Dict[str, Any]:
        """
        Merge outputs from all three Phase3 components into a single complete report.
        
        Args:
            out_dir: Phase3 output directory
            include_survey: Whether to include survey results
            include_contribution: Whether to include contribution analysis
            include_core_task: Whether to include core task comparisons
            
        Returns:
            Complete merged report dictionary
        """
        base_dir = Path(out_dir)
        phase3_dir = base_dir / "phase3"
        complete_report: Dict[str, Any] = {
            "original_paper": {},
            "core_task_survey": {},
            "contribution_analysis": {},
            "core_task_comparisons": {},
            "references": {},
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "pipeline_version": "1.0",
                "components": {
                    "survey": include_survey,
                    "contribution_analysis": include_contribution,
                    "core_task_comparisons": include_core_task,
                },
            },
        }
        
        # 1. Load original paper info (from Phase1)
        complete_report["original_paper"] = self._load_original_paper_info(base_dir)
        
        # 2. Load survey results
        survey_data: Dict[str, Any] = {}
        if include_survey:
            survey_data = self._load_survey_results(base_dir, phase3_dir)
            complete_report["core_task_survey"] = survey_data
        
        # 3. Load contribution analysis
        contrib_data: Dict[str, Any] = {}
        if include_contribution:
            contrib_data = self._load_contribution_analysis(phase3_dir)
            
            # CRITICAL: Apply audit policy to ensure data integrity even when loading from historical runs
            self._enforce_verified_refutation_policy(contrib_data)
            
            complete_report["contribution_analysis"] = contrib_data
            # If no real data was loaded, mark component as not-run while still
            # providing an empty shell for front-end consumers.
            if not contrib_data or (
                not contrib_data.get("contributions_with_results")
                and not contrib_data.get("contributions_no_results")
                and not contrib_data.get("statistics")
            ):
                complete_report["metadata"]["components"]["contribution_analysis"] = False
        
        # 3.5 Generate overall novelty assignment (if possible)
        if include_contribution and contrib_data:
            try:
                ona = self._generate_overall_novelty_assignment(base_dir, survey_data, contrib_data)
                if ona:
                    # Ensure overall_novelty_assignment appears at the BEGINNING
                    existing = complete_report.get("contribution_analysis") or {}
                    new_contrib = {"overall_novelty_assignment": ona}
                    new_contrib.update(existing)
                    complete_report["contribution_analysis"] = new_contrib
            except Exception as e:
                self.logger.warning(f"Failed to generate overall_novelty_assignment: {e}")
        
        # 4. Load core task comparisons
        core_task_data: Dict[str, Any] = {}
        if include_core_task:
            core_task_data = self._load_core_task_comparisons(phase3_dir)
            complete_report["core_task_comparisons"] = core_task_data
        
        # 5. Build references section from Phase2 citation_index.json (if available)
        try:
            references = self._build_references_section(base_dir)
            if references.get("items"):
                complete_report["references"] = references
            else:
                # If nothing to include, drop empty placeholder
                complete_report.pop("references", None)
        except Exception as e:
            self.logger.warning(f"Failed to build references section: {e}")
            complete_report.pop("references", None)

        # 6. Fill textual similarity segments into comparisons FIRST
        try:
            out_dir = Path(self.output_base_dir) / "phase3"
            similarity_file = out_dir / "textual_similarity_detection" / "results.json"
            
            if similarity_file.exists():
                self.logger.info(f"Loading textual similarity data to fill into comparisons...")
                similarity_data = json.loads(similarity_file.read_text(encoding='utf-8'))
                
                total_filled = 0
                
                # Fill into contribution_analysis comparisons
                if "contribution_analysis" in complete_report:
                    for contrib in complete_report["contribution_analysis"].get("contributions_with_results", []):
                        comparisons = contrib.get("comparisons", [])
                        if comparisons:
                            filled = self._fill_similarity_to_comparisons(comparisons, similarity_data)
                            total_filled += filled
                
                # Fill into core_task_comparisons
                if "core_task_comparisons" in complete_report:
                    comparisons = complete_report["core_task_comparisons"].get("comparisons", [])
                    if comparisons:
                        filled = self._fill_similarity_to_comparisons(comparisons, similarity_data)
                        total_filled += filled
                        
                        ctc_stats = complete_report["core_task_comparisons"].get("statistics", {})
                        if ctc_stats:
                            # Recalculate total_similarity_segments
                            new_total_segs = sum(
                                len(c.get("plagiarism_segments", []))
                                for c in comparisons
                            )
                            # Recalculate comparisons_with_similarity_warnings
                            new_warnings = len([
                                c for c in comparisons
                                if c.get("plagiarism_segments")
                            ])
                            ctc_stats["total_similarity_segments"] = new_total_segs
                            ctc_stats["comparisons_with_similarity_warnings"] = new_warnings
                            self.logger.debug(
                                f"Updated core_task_comparisons stats: "
                                f"total_similarity_segments={new_total_segs}, "
                                f"comparisons_with_similarity_warnings={new_warnings}"
                            )
                
                self.logger.info(
                    f"✓ Filled textual similarity segments into {total_filled} comparisons"
                )
        except Exception as e:
            self.logger.warning(f"Failed to fill textual similarity segments into comparisons: {e}")
        
        # 6.5. Build unified plagiarism detection index AFTER filling segments
        # Use complete_report data which now has segments filled in
        try:
            # Extract filled data from complete_report
            filled_contrib_data = complete_report.get("contribution_analysis", {})
            filled_core_task_data = complete_report.get("core_task_comparisons", {})
            
            similarity_index = self._build_textual_similarity_index(filled_contrib_data, filled_core_task_data)
            complete_report["plagiarism_detection"] = similarity_index
        except Exception as e:
            self.logger.warning(f"Failed to build plagiarism detection index: {e}")

        # 7. Pre-compute deterministic Phase4 artifact filenames (for front-end upload/pipeline linking)
        try:
            meta = complete_report.get("metadata") or {}
            artifacts = meta.get("artifacts") or {}
            artifacts["phase4_lightweight_pdf_filename"] = phase4_lightweight_pdf_filename_from_phase3_report(complete_report)
            artifacts["phase4_lightweight_md_filename"] = phase4_lightweight_md_filename_from_phase3_report(complete_report)
            meta["artifacts"] = artifacts
            complete_report["metadata"] = meta
        except Exception as e:
            self.logger.warning(f"Failed to compute Phase4 artifact filenames: {e}")
        
        # Save complete report
        complete_report_path = phase3_dir / "phase3_complete_report.json"
        with open(complete_report_path, "w", encoding="utf-8") as f:
            json.dump(complete_report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Generated complete Phase3 report: {complete_report_path}")
        return complete_report

    # ------------------------------------------------------------------------
    # Overall novelty assignment (Phase3-level summary)
    # ------------------------------------------------------------------------

    def _build_textual_similarity_index(
        self,
        contrib_data: Dict[str, Any],
        core_task_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a unified index of textual similarity segments.
        
        Source priority (for UI consumption):
        1) Prefer the dedicated similarity module (Part 3): phase3/textual_similarity_detection/results.json
           - Unified detection, deduplicated, complete
        2) Fallback: collect from comparison files (backward compatibility only)
           - In the current architecture, comparisons no longer emit similarity data, so this is usually empty
           - Kept to read older runs (pre-Part3 similarity split)

        This is intended for front-end consumption so that all similarity
        signals can be displayed from a single field in phase3_complete_report.
        """
        # Try loading from the independent detection module first (NEW)
        try:
            out_dir = Path(self.output_base_dir) / "phase3"
            similarity_file = out_dir / "textual_similarity_detection" / "results.json"
            
            if similarity_file.exists():
                self.logger.info(f"Loading textual similarity from {similarity_file}")
                import json
                similarity_data = json.loads(similarity_file.read_text(encoding='utf-8'))
                
                # The results.json already has the correct structure
                # Support both old (papers_with_similarities) and new (papers_with_plagiarism) keys
                papers_key = "papers_with_plagiarism" if "papers_with_plagiarism" in similarity_data else "papers_with_similarities"
                if isinstance(similarity_data, dict) and papers_key in similarity_data:
                    items = similarity_data.get(papers_key, [])
                    stats = similarity_data.get("statistics", {})
                    
                    if stats.get("is_skipped"):
                        self.logger.info("Textual similarity detection was skipped (placeholder data)")
                    
                    self.logger.info(
                        f"Loaded {len(items)} papers "
                        f"with similarities from independent detection module (key: {papers_key})"
                    )
                    return {
                        "statistics": stats,
                        "items": items
                    }
        except Exception as e:
            self.logger.warning(
                f"Failed to load textual similarity from independent module: {e}. "
                "Falling back to collecting from comparison results."
            )
        
        # Fallback: collect from comparison results (backward compatibility)
        # Note: with the current Part 3 pipeline, this usually returns empty
        # because Part 4/5 comparisons no longer generate similarity segments.
        self.logger.info("Using fallback: collecting textual similarity from comparison results")
        
        # Key: canonical_id or paper_id, Value: paper_entry
        aggregated: Dict[str, Dict[str, Any]] = {}
        total_unique_segments = 0

        # Helper to aggregate an entry
        def aggregate_entry(
            paper_id: str,
            title: str,
            url: str,
            source_label: str,
            segments: List[Dict[str, Any]]
        ) -> None:
            nonlocal total_unique_segments
            if not paper_id or not segments:
                return
            
            if paper_id not in aggregated:
                aggregated[paper_id] = {
                    "canonical_id": paper_id,
                    "title": title,
                    "url": url,
                    "sources": [],
                    "segments": [],
                    "_segment_hashes": set() # Internal for de-duplication
                }
            
            paper_entry = aggregated[paper_id]
            
            # Add source if unique
            if source_label not in paper_entry["sources"]:
                paper_entry["sources"].append(source_label)
            
            # Add segments if unique by content
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                orig_text = (seg.get("original_text") or "").strip()
                if not orig_text:
                    continue
                # Use a simple prefix-based hash or full content if not too long
                content_key = orig_text[:200] # Use first 200 chars as key
                if content_key not in paper_entry["_segment_hashes"]:
                    paper_entry["_segment_hashes"].add(content_key)
                    paper_entry["segments"].append(seg)
                    total_unique_segments += 1

        # 1) From core task comparisons
        try:
            core_comparisons = core_task_data.get("comparisons", []) or []
            for comp in core_comparisons:
                if not isinstance(comp, dict):
                    continue
                # Support new and legacy field names for backward compatibility
                segs = comp.get("plagiarism_segments") or comp.get("textual_similarity_segments") or comp.get("similarity_segments") or []
                if not segs:
                    continue
                
                paper_id = comp.get("canonical_id") or comp.get("retrieved_paper_id")
                aggregate_entry(
                    paper_id=paper_id,
                    title=comp.get("candidate_paper_title", "Unknown"),
                    url=comp.get("candidate_paper_url", ""),
                    source_label="Core Task Comparison",
                    segments=segs
                )
        except Exception as e:
            self.logger.debug(f"Failed to collect core-task similarity segments: {e}")

        # 2) From contribution analysis
        try:
            contribs_with_results = contrib_data.get("contributions_with_results", []) or []
            for contrib in contribs_with_results:
                if not isinstance(contrib, dict):
                    continue
                contrib_name = contrib.get("contribution_name", "Unknown Contribution")
                for comp in contrib.get("comparisons", []) or []:
                    if not isinstance(comp, dict):
                        continue
                    # Support new and legacy field names for backward compatibility
                    segs = comp.get("plagiarism_segments") or comp.get("textual_similarity_segments") or comp.get("similarity_segments") or []
                    if not segs:
                        continue
                    
                    paper_id = comp.get("canonical_id") or comp.get("paper_id")
                    aggregate_entry(
                        paper_id=paper_id,
                        title=comp.get("title", comp.get("candidate_paper_title", "Unknown")),
                        url=comp.get("url", comp.get("candidate_paper_url", "")),
                        source_label=f"Contribution: {contrib_name}",
                        segments=segs
                    )
        except Exception as e:
            self.logger.debug(f"Failed to collect contribution similarity segments: {e}")

        # Convert to final list and remove internal keys
        final_items = []
        for paper_id in sorted(aggregated.keys()):
            item = aggregated[paper_id]
            del item["_segment_hashes"]
            final_items.append(item)

        if not final_items:
            # Even when no segments are detected, emit a stable structure
            return {
                "statistics": {
                    "total_papers": 0,
                    "total_segments": 0,
                    "note_en": "No high-similarity text segments were detected across any compared papers.",
                },
                "items": [],
            }

        return {
            "statistics": {
                "total_papers": len(final_items),
                "total_segments": total_unique_segments,
                "note_en": "High-similarity text segments were detected for one or more compared papers.",
            },
            "items": final_items,
        }
    
    def _fill_similarity_to_comparisons(
        self,
        comparisons: List[Dict[str, Any]],
        similarity_data: Dict[str, Any]
    ) -> int:
        """
        Fill textual similarity segments into comparison entries.
        
        Args:
            comparisons: List of comparison dicts from contribution_analysis or core_task_comparisons
            similarity_data: Loaded data from textual_similarity_detection/results.json
            
        Returns:
            Number of comparisons filled with similarity data
        """
        if not comparisons or not similarity_data:
            return 0
        
        # Build canonical_id -> segments mapping
        similarity_map = {}
        
        # Support both old and new key names for backward compatibility
        papers_with_key = "papers_with_plagiarism" if "papers_with_plagiarism" in similarity_data else "papers_with_similarities"
        papers_without_key = "papers_without_plagiarism" if "papers_without_plagiarism" in similarity_data else "papers_without_similarities"
        
        # From papers_with_similarities/papers_with_plagiarism
        for paper in similarity_data.get(papers_with_key, []):
            canonical_id = paper.get("canonical_id")
            if canonical_id:
                # Support both "segments" and "plagiarism_segments" keys
                segments = paper.get("plagiarism_segments") or paper.get("segments", [])
                similarity_map[canonical_id] = segments
        
        # From papers_without_similarities/papers_without_plagiarism (empty segments)
        for paper in similarity_data.get(papers_without_key, []):
            canonical_id = paper.get("canonical_id")
            if canonical_id:
                similarity_map[canonical_id] = []
        
        # Fill into comparisons
        filled_count = 0
        for comp in comparisons:
            # Try multiple ID fields for compatibility
            paper_id = (
                comp.get("canonical_id") or 
                comp.get("paper_id") or 
                comp.get("retrieved_paper_id")
            )
            
            if paper_id in similarity_map:
                comp["plagiarism_segments"] = similarity_map[paper_id]
                filled_count += 1
                self.logger.debug(
                    f"Filled {len(similarity_map[paper_id])} plagiarism segments for {paper_id}"
                )
            elif "plagiarism_segments" not in comp:
                # Ensure field exists even if no data found
                comp["plagiarism_segments"] = []
        
        return filled_count

    def _generate_overall_novelty_assignment(
        self,
        base_dir: Path,
        survey_data: Dict[str, Any],
        contrib_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Generate an overall novelty summary for the contribution analysis section.

        Inputs (evidence sources):
        - Original paper abstract + (best-effort) introduction from Phase1.
        - Taxonomy narrative overview from the core-task survey (if available).
        - Per-contribution refutation Boolean flags (is_clearly_refuted), which ONLY
          indicate that at least one examined candidate appears to provide overlapping
          prior work for that contribution (they are NOT an exhaustive literature search).

        Output schema:
        - summary_paragraph: single 150–200 word paragraph of purely narrative text
          (no headings, lists, scores, or hard labels), explaining in a calm, neutral tone
          how the abstract+introduction, taxonomy narrative (if present), and
          contribution-level refutation flags jointly inform the perceived novelty
          of the work. This is intended to be used as the “Originality / Novelty”
          paragraph in Phase4, not as a final accept/reject verdict.
        """
        # Require contribution data
        contributions_with_results = contrib_data.get("contributions_with_results") or []
        if not isinstance(contributions_with_results, list) or not contributions_with_results:
            return None

        # Load Phase1 paper context
        phase1_dir = base_dir / "phase1"
        paper_json_path = phase1_dir / PAPER_JSON
        abstract_text = ""
        intro_text = ""
        try:
            if paper_json_path.exists():
                paper_data = json.loads(paper_json_path.read_text(encoding="utf-8"))
                abstract_text = (paper_data.get("abstract") or "").strip()
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            abstract_text = ""

        # Best-effort introduction from phase1_extracted.json
        phase1_extracted = phase1_dir / PHASE1_EXTRACTED_JSON
        try:
            if phase1_extracted.exists():
                extracted = json.loads(phase1_extracted.read_text(encoding="utf-8"))
                intro_candidate = ""

                # 1) Look for an explicit introduction section
                sections = extracted.get("sections")
                if isinstance(sections, list):
                    for sec in sections:
                        if not isinstance(sec, dict):
                            continue
                        title = (sec.get("title") or sec.get("name") or "").lower()
                        if "introduction" in title:
                            intro_candidate = (sec.get("text") or sec.get("content") or "").strip()
                            if intro_candidate:
                                break
                    # Fallback: first section text
                    if not intro_candidate and sections:
                        first_sec = sections[0]
                        if isinstance(first_sec, dict):
                            intro_candidate = (first_sec.get("text") or first_sec.get("content") or "").strip()

                # 2) Fallback: core_task.text
                if not intro_candidate:
                    core_task = extracted.get("core_task") or {}
                    if isinstance(core_task, dict):
                        intro_candidate = (core_task.get("text") or "").strip()

                intro_text = intro_candidate or ""
        except (json.JSONDecodeError, FileNotFoundError, IOError, TypeError, AttributeError):
            intro_text = ""

        # Truncate to reasonable length for prompt
        def _truncate(s: str, max_chars: int = 2000) -> str:
            s = (s or "").strip()
            return s[:max_chars]

        paper_context = {
            "abstract": _truncate(abstract_text, 1200),
            "introduction": _truncate(intro_text, 2000),
        }

        # Extract taxonomy tree and position information
        taxonomy_tree = {}
        original_paper_position = {}
        try:
            if survey_data:
                # Get full taxonomy tree
                taxonomy_tree = survey_data.get("taxonomy", {})
                
                # Get original paper position info
                highlight = survey_data.get("highlight", {})
                original_paper_id = highlight.get("original_paper_id")
                
                # Find the leaf containing the original paper
                mapping = survey_data.get("mapping", [])
                original_paper_leaf = None
                original_leaf_path = []
                sibling_paper_ids = []
                
                if original_paper_id and mapping:
                    for entry in mapping:
                        if entry.get("canonical_id") == original_paper_id:
                            original_leaf_path = entry.get("taxonomy_path", [])
                            if original_leaf_path:
                                original_paper_leaf = original_leaf_path[-1]
                            break
                    
                    # Find sibling papers in the same leaf
                    for entry in mapping:
                        if (entry.get("taxonomy_path", []) == original_leaf_path 
                            and entry.get("canonical_id") != original_paper_id):
                            sibling_paper_ids.append(entry.get("canonical_id"))
                
                # Get statistics
                stats = survey_data.get("statistics", {})
                original_paper_position = {
                    "canonical_id": original_paper_id,
                    "leaf_name": original_paper_leaf,
                    "leaf_path": original_leaf_path,
                    "sibling_paper_ids": sibling_paper_ids[:10],  # Limit to avoid overflow
                    "total_papers_in_taxonomy": stats.get("top50_count", 0),
                    "total_leaf_nodes": stats.get("leaf_count", 0),
                }
        except Exception as e:
            self.logger.warning(f"Failed to extract taxonomy position: {e}")
            taxonomy_tree = {}
            original_paper_position = {}

        # Get taxonomy narrative (complementary to the tree structure)
        taxonomy_narrative = ""
        try:
            if survey_data:
                taxonomy_narrative = (
                    survey_data.get("narrative", {}) or {}
                ).get("overview", "") or ""
        except Exception:
            taxonomy_narrative = ""

        # Extract literature search scope statistics
        literature_search_scope = {}
        try:
            stats_summary = contrib_data.get("statistics", {}).get("summary", {})
            literature_search_scope = {
                "note": "This analysis is based on a LIMITED literature search (top-K semantic search + citation expansion), not exhaustive.",
                "total_candidates_examined": stats_summary.get("total_candidates", 0),
                "total_refutable_pairs_found": stats_summary.get("total_refutable_pairs", 0),
                "total_contributions_analyzed": stats_summary.get("total_contributions_with_results", 0),
            }
        except Exception as e:
            self.logger.warning(f"Failed to extract search scope stats: {e}")
            literature_search_scope = {
                "note": "This analysis is based on a LIMITED literature search, not exhaustive."
            }

        # Aggregate per-contribution detailed statistics
        contrib_overview: List[Dict[str, Any]] = []
        try:
            stats_by_contrib = contrib_data.get("statistics", {}).get("by_contribution", {})
            
            for group in contributions_with_results:
                if not isinstance(group, dict):
                    continue
                name = group.get("contribution_name") or group.get("name") or ""
                if not name:
                    continue
                
                # Get detailed stats for this contribution
                contrib_stats = stats_by_contrib.get(name, {})
                comparisons = group.get("comparisons") or []
                any_can_refute = any(
                    isinstance(comp, dict) and comp.get("refutation_status") == "can_refute"
                    for comp in comparisons
                )
                
                contrib_overview.append({
                    "name": name,
                    "is_clearly_refuted": bool(any_can_refute),
                    "candidates_examined": contrib_stats.get("total_candidates", len(comparisons)),
                    "refutable_candidates": contrib_stats.get("refutable_candidates", 0),
                    "non_refutable_or_unclear": contrib_stats.get("non_refutable_or_unclear", 0),
                })
        except Exception as e:
            self.logger.warning(f"Failed to build contribution overview with stats: {e}")
            # Fallback to basic boolean-only version
            for group in contributions_with_results:
                if not isinstance(group, dict):
                    continue
                name = group.get("contribution_name") or group.get("name") or ""
                if not name:
                    continue
                comparisons = group.get("comparisons") or []
                any_can_refute = any(
                    isinstance(comp, dict) and comp.get("refutation_status") == "can_refute"
                    for comp in comparisons
                )
                contrib_overview.append({
                    "name": name,
                    "is_clearly_refuted": bool(any_can_refute),
                })

        if not contrib_overview and not taxonomy_tree and not taxonomy_narrative and not any(paper_context.values()):
            # Nothing to condition on
            return None

        # Prepare LLM client
        try:
            llm_client = create_llm_client()
        except Exception as e:
            self.logger.warning(f"Failed to create LLM client for overall_novelty_assignment: {e}")
            llm_client = None

        if not llm_client:
            return None

        # Build prompts
        sys_prompt = (
            "You are helping a human reviewer write the 'Originality / Novelty' assessment in a review form.\n"
            "You are not deciding accept/reject; you are explaining how you read the paper's novelty in context.\n\n"
            "You will receive FIVE kinds of signal:\n"
            "1. The paper's abstract and a best-effort introduction (may be incomplete or imperfect).\n"
            "2. A complete taxonomy tree showing the hierarchical structure of the research field (50 papers across ~36 topics).\n"
            "3. The original paper's position in this taxonomy (which leaf it belongs to, what sibling papers are in that leaf).\n"
            "4. Literature search scope: how many candidate papers were examined (e.g., 30 papers from top-K semantic search).\n"
            "5. Per-contribution analysis results with detailed statistics (e.g., Contribution A: examined 10 papers, 1 can refute).\n\n"
            "CRITICAL CONTEXT:\n"
            "- The taxonomy tree provides COMPLETE field structure: you can see how crowded or sparse each research direction is.\n"
            "- The 'sibling papers' in the same taxonomy leaf as the original paper are the MOST relevant prior work.\n"
            "- The statistics tell you the SCALE of the literature search (e.g., 30 candidates examined, not 300).\n"
            "- 'is_clearly_refuted' ONLY means that among the LIMITED candidates examined, at least one appears to provide\n"
            "  overlapping prior work. It does NOT mean the literature search was exhaustive.\n"
            "- Use the taxonomy tree to assess whether the paper sits in a crowded vs. sparse research area.\n\n"
            "Your job is to write 3-4 SHORT paragraphs that describe how novel the work feels given these signals.\n"
            "Write as a careful, neutral reviewer. Each paragraph should focus on ONE specific aspect:\n\n"
            "PARAGRAPH 1 (50-70 words): Core Contribution & Taxonomy Position\n"
            "- Summarize what the paper contributes.\n"
            "- Use the taxonomy tree to explain where it sits: which leaf? How many sibling papers in that leaf?\n"
            "- Is this a crowded or sparse research direction?\n\n"
            "PARAGRAPH 2 (50-70 words): Field Context & Neighboring Work\n"
            "- Based on the taxonomy tree structure, identify nearby leaves and branches.\n"
            "- Which related directions exist? How does this work connect to or diverge from them?\n"
            "- Use the scope_note and exclude_note from taxonomy nodes to clarify boundaries.\n\n"
            "PARAGRAPH 3 (50-70 words): Prior Work Overlap & Literature Search Scope\n"
            "- Discuss the contribution-level statistics: e.g., 'Contribution A examined 10 candidates, 1 can refute'.\n"
            "- Be explicit about the search scale: 'Among 30 candidates examined...' not 'prior work shows...'.\n"
            "- Which contributions appear more novel? Which have more substantial prior work?\n\n"
            "PARAGRAPH 4 (40-60 words, OPTIONAL): Overall Assessment & Limitations\n"
            "- Brief synthesis of your impression given the LIMITED search scope.\n"
            "- Acknowledge what the analysis covers and what it doesn't (e.g., 'based on top-30 semantic matches').\n"
            "- Only include if you have substantive points; omit if redundant.\n\n"
            "FORM REQUIREMENTS:\n"
            "- Each paragraph must be a separate string in the output array (separated by \\n\\n when rendered).\n"
            "- Do NOT use headings, bullet points, or numbered lists within paragraphs.\n"
            "- Do not output numeric scores, probabilities, grades, or hard counts of papers/contributions.\n"
            "- Avoid blunt verdicts ('clearly not novel', 'definitively incremental'); keep the tone descriptive and analytic.\n"
            "- Write in English only.\n\n"
            "Return STRICT JSON only (no code fences) with the following schema:\n"
            "{\n"
            "  'paragraphs': [\n"
            "    'Paragraph 1 text (50-70 words)',\n"
            "    'Paragraph 2 text (50-70 words)',\n"
            "    'Paragraph 3 text (50-70 words)',\n"
            "    'Paragraph 4 text (40-60 words, optional)'\n"
            "  ]\n"
            "}\n"
        )

        user_payload = {
            "paper_context": paper_context,
            
            # NEW: Complete taxonomy tree (provides full field structure)
            "taxonomy_tree": taxonomy_tree,
            
            # NEW: Original paper's position in taxonomy
            "original_paper_position": original_paper_position,
            
            # NEW: Literature search scope and statistics
            "literature_search_scope": literature_search_scope,
            
            # ENHANCED: Contributions with detailed statistics
            "contributions": contrib_overview,
            
            # KEPT: Narrative as complementary natural language context
            "taxonomy_narrative": taxonomy_narrative,
        }

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        # Primary attempt: JSON mode for structured output
        # Increased max_tokens to accommodate 3-4 paragraphs (~200-280 words)
        try:
            result = llm_client.generate_json(messages, max_tokens=600)  # type: ignore[assignment]
        except Exception as e:
            self.logger.warning(f"LLM call failed for overall_novelty_assignment: {e}")
            result = None

        paragraphs: List[str] = []

        if isinstance(result, dict):
            # Extract paragraphs array from JSON response
            raw_paragraphs = result.get("paragraphs", [])
            if isinstance(raw_paragraphs, list):
                paragraphs = [str(p).strip() for p in raw_paragraphs if p and str(p).strip()]
        else:
            # Fallback path: use plain-text generation + light parsing.
            self.logger.warning(
                "overall_novelty_assignment LLM result is not a JSON object; "
                "falling back to text mode + manual parsing."
            )
            try:
                raw_text = llm_client.generate(messages, max_tokens=600)  # type: ignore[arg-type]
            except TypeError:
                raw_text = llm_client.generate(messages, max_tokens=600)  # type: ignore[call-arg]
            except Exception as e:
                self.logger.warning(f"Text-mode fallback failed for overall_novelty_assignment: {e}")
                raw_text = ""

            raw_text = (raw_text or "").strip()
            if raw_text:
                # 1) Try to interpret as Python dict literal: {'paragraphs': [...]}
                try:
                    import ast

                    obj = ast.literal_eval(raw_text)
                    if isinstance(obj, dict) and isinstance(obj.get("paragraphs"), list):
                        paragraphs = [str(p).strip() for p in obj["paragraphs"] if p and str(p).strip()]
                except Exception:
                    pass

                # 2) If still empty, try JSON parsing directly
                if not paragraphs:
                    try:
                        obj = json.loads(raw_text)
                        if isinstance(obj, dict) and isinstance(obj.get("paragraphs"), list):
                            paragraphs = [str(p).strip() for p in obj["paragraphs"] if p and str(p).strip()]
                    except Exception:
                        pass

                # 3) As a last resort, split the text by double newlines
                if not paragraphs and raw_text:
                    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

        if not paragraphs:
            # Require at least some narrative
            return None

        # Return both the paragraphs array and a combined summary_paragraph for backward compatibility
        combined_text = "\n\n".join(paragraphs)
        ona: Dict[str, Any] = {
            "paragraphs": paragraphs,
            "summary_paragraph": combined_text,  # For backward compatibility
        }
        return ona
    
    def _load_original_paper_info(self, base_dir: Path) -> Dict[str, Any]:
        """
        Load original paper info from Phase2 citation_index (canonical source).
        Phase2 is the single source of truth for canonical_id.
        """
        paper_info = {}
        
        # Priority: Read from Phase2 citation_index.json (index 0 = original paper)
        phase2_citation_index = base_dir / "phase2" / "final" / "citation_index.json"
        if phase2_citation_index.exists():
            try:
                citation_data = json.loads(phase2_citation_index.read_text(encoding="utf-8"))
                items = citation_data.get("items", [])
                if items and len(items) > 0:
                    original_paper = items[0]  # Index 0 is always the original paper
                    # Verify it's the original paper
                    roles = original_paper.get("roles", [])
                    if any(r.get("type") == "original_paper" for r in roles):
                        paper_info = {
                            "canonical_id": original_paper.get("canonical_id", ""),
                            "title": original_paper.get("title", ""),
                            "abstract": original_paper.get("abstract", ""),
                            "authors": original_paper.get("authors", []),
                            "venue": original_paper.get("venue", ""),
                            "year": original_paper.get("year"),
                            "doi": original_paper.get("doi"),
                            "arxiv_id": original_paper.get("arxiv_id"),
                            "openreview_id": original_paper.get("openreview_id"),
                            "url": original_paper.get("url", ""),
                        }
                        self.logger.info(f"Loaded original paper from Phase2 citation_index: {paper_info['canonical_id']}")
                        
                        # Load additional metadata from Phase1 if available (keywords, primary_area, etc.)
                        phase1_paper = base_dir / "phase1" / PAPER_JSON
                        if phase1_paper.exists():
                            try:
                                phase1_data = json.loads(phase1_paper.read_text(encoding="utf-8"))
                                paper_info.setdefault("keywords", phase1_data.get("keywords"))
                                paper_info.setdefault("primary_area", phase1_data.get("primary_area", ""))
                                paper_info.setdefault("openreview_rating_mean", phase1_data.get("openreview_rating_mean"))
                            except Exception as e:
                                self.logger.debug(f"Could not load Phase1 metadata: {e}")
                        
                        return paper_info
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                self.logger.warning(f"Failed to load from Phase2 citation_index: {e}")
        
        # Fallback: Read from Phase1 paper.json (for backward compatibility)
        # This should not normally be used if Phase2 has been run
        phase1_dir = base_dir / "phase1"
        if (phase1_dir / PAPER_JSON).exists():
            try:
                paper_data = json.loads((phase1_dir / PAPER_JSON).read_text(encoding="utf-8"))
                # Generate canonical_id using current strategy (title hash)
                from paper_novelty_pipeline.utils.paper_id import make_canonical_id
                canonical_id = make_canonical_id(
                    title=paper_data.get("title"),
                    paper_id=paper_data.get("paper_id"),
                    url=paper_data.get("original_pdf_url") or paper_data.get("paper_id"),
                )
                paper_info = {
                    "canonical_id": canonical_id,
                    "title": paper_data.get("title", ""),
                    "abstract": paper_data.get("abstract", ""),
                    "authors": paper_data.get("authors", []),
                    "venue": paper_data.get("venue", ""),
                    "year": paper_data.get("year"),
                    "doi": paper_data.get("doi"),
                    "arxiv_id": paper_data.get("arxiv_id"),
                    "openreview_id": paper_data.get("openreview_id") or paper_data.get("openreview_forum_id"),
                    "keywords": paper_data.get("keywords"),
                    "primary_area": paper_data.get("primary_area", ""),
                    "openreview_rating_mean": paper_data.get("openreview_rating_mean"),
                }
                self.logger.warning(f"Loaded from Phase1 (fallback). Regenerated canonical_id: {canonical_id}")
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                self.logger.warning(f"Failed to load original paper info from Phase1: {e}")
        
        return paper_info
    
    def _load_survey_results(
        self,
        base_dir: Path,
        phase3_dir: Path,
    ) -> Dict[str, Any]:
        """Load survey results from files."""
        survey_dir = phase3_dir / "core_task_survey"
        survey_data = {}
        
        try:
            # Try new location first
            survey_report_path = survey_dir / "survey_report.json"
            if not survey_report_path.exists():
                survey_report_path = phase3_dir / "json_report.json"  # Fallback to old location
            
            if survey_report_path.exists():
                survey_report = json.loads(survey_report_path.read_text(encoding="utf-8"))
                short_survey = survey_report.get("short_survey", {})
                
                # Load taxonomy
                taxonomy = {}
                if (survey_dir / "taxonomy.json").exists():
                    taxonomy = json.loads((survey_dir / "taxonomy.json").read_text(encoding="utf-8"))
                else:
                    taxonomy = short_survey.get("taxonomy", {})
                
                # Load narrative
                narrative_data = {}
                overview_text = ""
                
                # Priority 1: narrative.json (newer format)
                if (survey_dir / "narrative.json").exists():
                    narrative_file_data = json.loads(
                        (survey_dir / "narrative.json").read_text(encoding="utf-8")
                    )
                    # Newer writers store the main text under 'narrative', older ones under 'overview'
                    overview_text = (
                        narrative_file_data.get("overview")
                        or narrative_file_data.get("narrative")
                        or ""
                    )
                else:
                    # Priority 2: survey_report.narrative_overview.markdown
                    narrative_overview = survey_report.get("short_survey", {}).get("narrative_overview", {})
                    if narrative_overview and "markdown" in narrative_overview:
                        overview_text = narrative_overview.get("markdown", "")
                    else:
                        # Priority 3: survey_report.narrative.overview
                        narrative_raw = short_survey.get("narrative", {})
                        overview_text = narrative_raw.get("overview", "")
                
                narrative_data = {"overview": overview_text}
                
                # Load statistics
                # Primary source: legacy meta.stats (top50_count, unique_count, duplicate_groups)
                statistics = survey_report.get("meta", {}).get("stats", {}) or {}
                # Newer writers may store tree_depth / leaf_count etc. under
                # short_survey.statistics; merge them in (short_survey wins).
                short_stats = short_survey.get("statistics", {}) or {}
                if short_stats:
                    try:
                        merged = dict(statistics)
                        merged.update(short_stats)
                        statistics = merged
                    except Exception:
                        # If anything goes wrong during merge, fall back to legacy stats
                        pass
                
                # Bug fix: Recalculate leaf_count based on actual taxonomy structure
                # (legacy stats may not match the actual taxonomy)
                if taxonomy:
                    def _count_leaves(node: dict) -> int:
                        children = node.get("children", [])
                        if not children:
                            return 1
                        return sum(_count_leaves(c) for c in children)
                    
                    actual_leaf_count = _count_leaves(taxonomy)
                    if statistics.get("leaf_count") != actual_leaf_count:
                        self.logger.debug(
                            f"Correcting leaf_count: {statistics.get('leaf_count')} -> {actual_leaf_count}"
                        )
                        statistics["leaf_count"] = actual_leaf_count
                
                # Get display_index and highlight
                display_index = short_survey.get("display_index", {})
                highlight = short_survey.get("highlight", {})
                
                # Get taxonomy_diagnostics (if available)
                taxonomy_diagnostics = short_survey.get("taxonomy_diagnostics", {})
                
                # Rebuild mapping from taxonomy
                mapping = self._rebuild_mapping_from_taxonomy(taxonomy) if taxonomy else short_survey.get("mapping", [])
                
                if taxonomy:
                    self.logger.info(f"Rebuilt mapping from taxonomy with {len(mapping)} entries")
                else:
                    self.logger.warning(
                        "Taxonomy not found, using mapping from survey_report.json "
                        "(may contain outdated taxonomy names)"
                    )
                
                # Build papers index from Phase2 Top50
                papers_index = self._build_papers_index(
                    base_dir, mapping, display_index, highlight, survey_report.get("paper_info", {})
                )
                
                # Optionally load textual taxonomy tree from markdown report (Scheme A)
                text_tree_markdown = ""
                markdown_report_path = survey_dir / "reports" / "core_task_survey.md"
                try:
                    if markdown_report_path.exists():
                        lines = markdown_report_path.read_text(encoding="utf-8").splitlines()
                        in_taxonomy_block = False
                        collected_lines = []
                        for line in lines:
                            stripped = line.strip()
                            if not in_taxonomy_block:
                                # Start at "## Taxonomy (text)" heading
                                if stripped.lower().startswith("## taxonomy"):
                                    in_taxonomy_block = True
                                    collected_lines.append(line)
                            else:
                                # Stop when we hit the next top-level section (another '## ' heading)
                                if stripped.startswith("## ") and not stripped.lower().startswith("## taxonomy"):
                                    break
                                collected_lines.append(line)
                        text_tree_markdown = "\n".join(collected_lines).strip()
                except (OSError, IOError, UnicodeDecodeError) as e:
                    self.logger.warning(f"Failed to load text_tree_markdown from markdown: {e}")
                
                survey_data = {
                    "statistics": statistics,
                    "taxonomy": taxonomy,
                    "papers": papers_index,
                    "mapping": mapping,
                    "display_index": display_index,
                    "highlight": {
                        "original_paper_id": highlight.get("original_paper_id", "")
                    } if highlight else {},
                    "narrative": narrative_data,
                    "text_tree_markdown": text_tree_markdown if text_tree_markdown else None,
                    "taxonomy_diagnostics": taxonomy_diagnostics,
                }
            else:
                self.logger.warning("Survey report not found")
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(f"Failed to load survey results: {e}")
        
        return survey_data
    
    def _rebuild_mapping_from_taxonomy(
        self,
        taxo: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Rebuild mapping from taxonomy tree to ensure correct taxonomy names in paths.
        
        Logic consistent with phase3_survey.py _collect_leaf_assignments.
        """
        mapping_dict: Dict[str, List[str]] = {}
        
        def collect_leaf_assignments(node: Dict[str, Any], path: List[str]) -> None:
            """Recursively collect all leaf node paper_ids and their taxonomy paths."""
            subs = node.get("subtopics", [])
            papers = node.get("papers", [])
            name = node.get("name", "")
            
            if isinstance(subs, list) and subs:
                # Internal node: continue recursion, add current name to path
                for child in subs:
                    if isinstance(child, dict):
                        collect_leaf_assignments(child, path + [name])
            elif isinstance(papers, list) and papers:
                # Leaf node: collect paper_id and path (including leaf name)
                leaf_path = path + [name]
                for pid in papers:
                    if isinstance(pid, str):
                        mapping_dict[pid] = leaf_path
        
        # Start from root (empty path, root name added in first recursion)
        collect_leaf_assignments(taxo, [])
        
        # Convert to list format
        return [
            {"canonical_id": pid, "taxonomy_path": path}
            for pid, path in mapping_dict.items()
        ]
    
    def _build_papers_index(
        self,
        base_dir: Path,
        mapping: List[Dict[str, Any]],
        display_index: Dict[str, int],
        highlight: Dict[str, Any],
        paper_info: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build papers index from Phase2 Top50."""
        papers_index = {}
        phase2_dir = base_dir / "phase2"
        core_task_file = phase2_dir / "final" / "core_task_perfect_top50.json"
        original_paper_id = paper_info.get("canonical_id", "")
        
        if core_task_file.exists():
            try:
                top50_items = read_json_list(core_task_file)
                
                # Helper: Generate all possible IDs for a paper
                def generate_all_possible_ids(paper_item: Dict[str, Any]) -> List[str]:
                    """Generate all possible ID formats for a paper."""
                    possible_ids = []
                    canonical_id = paper_item.get("canonical_id", "")
                    title = paper_item.get("title", "")
                    arxiv_id = paper_item.get("arxiv_id") or paper_item.get("raw_metadata", {}).get("arxiv_id")
                    doi = paper_item.get("doi") or paper_item.get("raw_metadata", {}).get("doi")
                    
                    # Add canonical_id as primary ID
                    if canonical_id:
                        possible_ids.append(canonical_id)
                    
                    # Add title-based ID
                    if title:
                        title_normalized = re.sub(r'\s+', ' ', title.lower().strip())
                        possible_ids.append(f"title:{title_normalized}")
                    
                    # Add arxiv ID
                    if arxiv_id:
                        if arxiv_id.startswith("arxiv:"):
                            possible_ids.append(arxiv_id)
                        else:
                            possible_ids.append(f"arxiv:{arxiv_id}")
                    
                    # Add DOI
                    if doi:
                        if doi.startswith("doi:"):
                            possible_ids.append(doi)
                        elif doi.startswith("10."):
                            possible_ids.append(f"doi:{doi}")
                        else:
                            possible_ids.append(f"doi:10.48550/arxiv.{doi}")
                    
                    return possible_ids
                
                # Build reverse mapping: canonical_id -> taxonomy_id
                taxonomy_id_map: Dict[str, str] = {}
                for map_entry in mapping:
                    canonical_id_in_tax = map_entry.get("canonical_id", "")
                    if canonical_id_in_tax:
                        # Map all possible IDs to this taxonomy_id
                        for pid in generate_all_possible_ids({"canonical_id": canonical_id_in_tax}):
                            taxonomy_id_map[pid] = canonical_id_in_tax
                
                # Process each Top50 paper
                for rank, paper_item in enumerate(top50_items, start=1):
                    # Try to find the taxonomy_id used in taxonomy/display_index
                    possible_ids = generate_all_possible_ids(paper_item)
                    taxonomy_id = None
                    
                    # First, check if any of our possible_ids match a taxonomy_id in display_index
                    for pid in possible_ids:
                        if pid in display_index:
                            taxonomy_id = pid
                            break
                    
                    # If not found, check mapping
                    if not taxonomy_id:
                        for pid in possible_ids:
                            if pid in taxonomy_id_map:
                                taxonomy_id = taxonomy_id_map[pid]
                                break
                    
                    # Fallback: use canonical_id
                    if not taxonomy_id:
                        taxonomy_id = paper_item.get("canonical_id", "")
                    
                    # Extract paper metadata
                    raw_meta = paper_item.get("raw_metadata") or {}
                    
                    # Extract authors using utils function
                    authors = extract_authors(paper_item)
                    
                    # Build paper entry
                    paper_entry = {
                        "canonical_id": paper_item.get("canonical_id", ""),
                        "title": paper_item.get("title", ""),
                        "authors": authors,
                        "abstract": paper_item.get("abstract", ""),
                        "year": paper_item.get("year"),
                        "venue": paper_item.get("venue"),
                        "url": paper_item.get("url") or paper_item.get("source_url", ""),
                        "doi": paper_item.get("doi"),
                        "arxiv_id": paper_item.get("arxiv_id"),
                        "is_original": (taxonomy_id == highlight.get("original_paper_id")) or any(
                            pid == original_paper_id for pid in possible_ids
                        ),
                        "display_index": display_index.get(taxonomy_id),
                        "relevance_score": paper_item.get("relevance_score"),
                        "rank": rank,
                    }
                    
                    # Use taxonomy_id as key
                    papers_index[taxonomy_id] = paper_entry
                
                self.logger.info(f"Built papers index with {len(papers_index)} papers")
            except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
                self.logger.warning(f"Failed to build papers index from Phase2: {e}")
        
        return papers_index
    
    def _load_contribution_analysis(
        self,
        phase3_dir: Path,
    ) -> Dict[str, Any]:
        """Load contribution analysis results."""
        # Default empty shell so the section always has a predictable shape
        contribution_data: Dict[str, Any] = {
            "statistics": {},
            "contributions_with_results": [],
            "contributions_no_results": [],
        }
        
        try:
            # Try to load from contribution_analysis/phase3_report.json (new location)
            contribution_analysis_dir = phase3_dir / "contribution_analysis"
            phase3_report_path = contribution_analysis_dir / "phase3_report.json"
            # Fallback to old location for backward compatibility
            if not phase3_report_path.exists():
                phase3_report_path = phase3_dir / "phase3_report.json"
            
            if phase3_report_path.exists():
                phase3_report = json.loads(phase3_report_path.read_text(encoding="utf-8"))
                
                # Migrate legacy field names in loaded data
                for contrib in phase3_report.get("contributions_with_results", []) or []:
                    for comp in contrib.get("comparisons", []) or []:
                        if isinstance(comp, dict) and "textual_similarity_segments" in comp and "plagiarism_segments" not in comp:
                            comp["plagiarism_segments"] = comp.pop("textual_similarity_segments")
                
                contribution_data = {
                    "statistics": phase3_report.get("statistics", {}) or {},
                    "contributions_with_results": phase3_report.get("contributions_with_results", []) or [],
                    "contributions_no_results": phase3_report.get("contributions_no_results", []) or [],
                }
            else:
                # Try to reconstruct from contribution_analysis directory
                contrib_dir = contribution_analysis_dir
                if contrib_dir.exists():
                    contributions_list: List[Dict[str, Any]] = []
                    total_candidates = 0
                    total_refutable_pairs = 0
                    by_contribution: Dict[str, Dict[str, Any]] = {}

                    for contrib_subdir in sorted(contrib_dir.iterdir()):
                        if not (
                            contrib_subdir.is_dir()
                            and contrib_subdir.name.startswith("contribution_")
                        ):
                            continue

                        meta_file = contrib_subdir / "contribution_meta.json"
                        if not meta_file.exists():
                            continue

                        try:
                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        except (json.JSONDecodeError, FileNotFoundError, IOError):
                            continue

                        comparisons: List[Dict[str, Any]] = []
                        for paper_file in contrib_subdir.glob("paper_*.json"):
                            try:
                                comp = json.loads(paper_file.read_text(encoding="utf-8"))
                                # Migrate legacy field name: textual_similarity_segments -> plagiarism_segments
                                if "textual_similarity_segments" in comp and "plagiarism_segments" not in comp:
                                    comp["plagiarism_segments"] = comp.pop("textual_similarity_segments")
                                comparisons.append(comp)
                            except (json.JSONDecodeError, FileNotFoundError, IOError):
                                continue

                        # Bug fix: Generate contribution_id from index or directory name
                        # (was None previously, breaking frontend identification)
                        contrib_index = meta.get("contribution_index")
                        contrib_id = f"contribution_{contrib_index}" if contrib_index is not None else contrib_subdir.name
                        
                        contrib_entry = {
                            "contribution_id": contrib_id,
                            "contribution_index": contrib_index,
                            "contribution_name": meta.get("contribution_name"),
                            "contribution_claim": meta.get("author_claim_text", ""),  # Also add claim text
                            "contribution_description": meta.get("description"),
                            "comparisons": comparisons,
                        }
                        contributions_list.append(contrib_entry)

                        # Update simple aggregate statistics per contribution
                        name = meta.get("contribution_name") or f"contribution_{meta.get('contribution_index')}"
                        total = len(comparisons)
                        total_candidates += total
                        refutable = sum(
                            1
                            for c in comparisons
                            if isinstance(c, dict) and c.get("refutation_status") == "can_refute"
                        )
                        total_refutable_pairs += refutable
                        by_contribution[name] = {
                            "total_candidates": total,
                            "refutable_candidates": refutable,
                            "non_refutable_or_unclear": max(0, total - refutable),
                        }

                    statistics: Dict[str, Any] = {
                        "summary": {
                            "total_contributions_with_results": len(contributions_list),
                            "total_candidates": total_candidates,
                            "total_refutable_pairs": total_refutable_pairs,
                        },
                        "by_contribution": by_contribution,
                    }

                    contribution_data = {
                        "statistics": statistics,
                        "contributions_with_results": contributions_list,
                        "contributions_no_results": [],
                    }
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(f"Failed to load contribution analysis: {e}")
        
        # Enforce a strict, schema-preserving policy:
        # - A candidate is only "can_refute" if it contains at least one evidence pair
        #   where BOTH sides were verified by EvidenceVerifier (found==true).
        # - Otherwise, force "cannot_refute" and drop unverified evidence from the report.
        try:
            self._enforce_verified_refutation_policy(contribution_data)
        except Exception as e:
            self.logger.warning(f"Failed to enforce verified-refutation policy: {e}")

        return contribution_data

    def _enforce_verified_refutation_policy(self, contribution_data: Dict[str, Any]) -> None:
        """
        Mutate contribution_data in-place to ensure refutation claims are backed by
        at least one fully verified evidence pair (both sides found==true).

        This is intentionally schema-preserving:
        - It only changes values (refutation_status) and prunes evidence arrays.
        - It does not require rerunning LLM calls.
        """
        if not isinstance(contribution_data, dict):
            return
        groups = contribution_data.get("contributions_with_results") or []
        if not isinstance(groups, list) or not groups:
            return

        def _is_verified_pair(pair: Any) -> bool:
            if not isinstance(pair, dict):
                return False
            ol = pair.get("original_location") or {}
            cl = pair.get("candidate_location") or {}
            if not isinstance(ol, dict) or not isinstance(cl, dict):
                return False
            return (ol.get("found") is True) and (cl.get("found") is True)

        def _get_all_pairs(comp: Dict[str, Any]) -> List[Dict[str, Any]]:
            # Prefer refutation_evidence.evidence_pairs when present, else fall back to legacy novelty_evidence
            ref_ev = comp.get("refutation_evidence")
            if isinstance(ref_ev, dict):
                pairs = ref_ev.get("evidence_pairs") or []
                if isinstance(pairs, list):
                    return [p for p in pairs if isinstance(p, dict)]
            pairs2 = comp.get("novelty_evidence") or []
            if isinstance(pairs2, list):
                return [p for p in pairs2 if isinstance(p, dict)]
            return []

        total_candidates = 0
        total_refutable_pairs = 0
        total_downgraded = 0

        stats = contribution_data.get("statistics") or {}
        if not isinstance(stats, dict):
            stats = {}
            contribution_data["statistics"] = stats
        by_contribution = stats.get("by_contribution") or {}
        if not isinstance(by_contribution, dict):
            by_contribution = {}
            stats["by_contribution"] = by_contribution

        for group in groups:
            if not isinstance(group, dict):
                continue
            comparisons = group.get("comparisons") or []
            if not isinstance(comparisons, list):
                continue

            contrib_name = group.get("contribution_name") or group.get("name") or ""
            total_candidates += len(comparisons)

            refutable_candidates = 0

            for comp in comparisons:
                if not isinstance(comp, dict):
                    continue

                all_pairs = _get_all_pairs(comp)
                verified_pairs = [p for p in all_pairs if _is_verified_pair(p)]

                if not verified_pairs:
                    # Final Audit Failure: if it's still 'can_refute' here, it means all retries failed.
                    was_originally_can_refute = comp.get("refutation_status") == "can_refute"
                    
                    comp["refutation_status"] = "cannot_refute"
                    
                    if was_originally_can_refute:
                        # This is a record of the final failure after retries
                        comp["brief_note"] = (
                            "[Final Audit Failure] The model insisted on a refutation claim but failed to provide "
                            "verifiable evidence after multiple retries. Marked as cannot_refute for safety. "
                            "Please manually verify the candidate text."
                        )
                        comp["verification_warning"] = True
                        total_downgraded += 1
                    
                    # Drop refutation_evidence to avoid carrying unverified content.
                    comp.pop("refutation_evidence", None)
                else:
                    # Verified Success: at least one pair is confirmed.
                    comp["refutation_status"] = "can_refute"
                    # Ensure a refutation_evidence object exists for can_refute rows.
                    ref_ev = comp.get("refutation_evidence")
                    if not isinstance(ref_ev, dict):
                        ref_ev = {
                            "summary": "",
                            "evidence_pairs": [],
                        }
                        comp["refutation_evidence"] = ref_ev
                    ref_ev["evidence_pairs"] = verified_pairs
                    refutable_candidates += 1

            total_refutable_pairs += refutable_candidates

            # Update per-contribution statistics if present (or create minimal record)
            if contrib_name:
                rec = by_contribution.get(contrib_name) or {}
                if not isinstance(rec, dict):
                    rec = {}
                rec["total_candidates"] = len(comparisons)
                rec["refutable_candidates"] = refutable_candidates
                rec["non_refutable_or_unclear"] = max(0, len(comparisons) - refutable_candidates)
                by_contribution[contrib_name] = rec

        summary = stats.get("summary") or {}
        if not isinstance(summary, dict):
            summary = {}
        summary["total_candidates"] = total_candidates
        summary["total_refutable_pairs"] = total_refutable_pairs
        
        # Update comparison stats with total_downgraded
        comp_stats = stats.get("comparison") or {}
        if isinstance(comp_stats, dict):
            comp_stats["refutations_downgraded_by_audit"] = total_downgraded
            stats["comparison"] = comp_stats

        # Keep other summary fields if present; set a sensible default for contributions_with_results.
        summary.setdefault("total_contributions_with_results", len([g for g in groups if isinstance(g, dict)]))
        stats["summary"] = summary
    
    def _load_core_task_comparisons(
        self,
        phase3_dir: Path,
    ) -> Dict[str, Any]:
        """Load core task comparison results."""
        comparisons_data = {}
        
        try:
            comparisons_dir = phase3_dir / "core_task_comparisons"
            
            # Load merged comparisons file
            merged_file = comparisons_dir / "core_task_comparisons.json"
            if merged_file.exists():
                merged_data = json.loads(merged_file.read_text(encoding="utf-8"))
                
                # Migrate legacy field names in loaded comparisons
                for comp in merged_data.get("comparisons", []) or []:
                    if isinstance(comp, dict) and "textual_similarity_segments" in comp and "plagiarism_segments" not in comp:
                        comp["plagiarism_segments"] = comp.pop("textual_similarity_segments")
                
                comparisons_data = {
                    "statistics": {},
                    "comparisons": merged_data.get("comparisons", []),
                }
            else:
                comparisons_data["comparisons"] = []
            
            # Load summary for statistics
            summary_file = comparisons_dir / "summary.json"
            if summary_file.exists():
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                comparisons_data["statistics"] = summary_data.get("statistics", {})
                comparisons_data["structural_position"] = summary_data.get("structural_position", {})
                subtopic_file = summary_data.get("subtopic_comparison_file")
                if subtopic_file:
                    comparisons_data["subtopic_comparison_file"] = subtopic_file
                    subtopic_path = comparisons_dir / subtopic_file
                    if subtopic_path.exists():
                        try:
                            comparisons_data["subtopic_comparison"] = json.loads(
                                subtopic_path.read_text(encoding="utf-8")
                            )
                        except Exception as e:
                            self.logger.warning(f"Failed to load subtopic_comparison: {e}")
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(f"Failed to load core task comparisons: {e}")
        
        return comparisons_data

    # ------------------------------------------------------------------------
    # References section (Phase2 citation_index.json passthrough)
    # ------------------------------------------------------------------------

    def _build_references_section(
        self,
        base_dir: Path,
    ) -> Dict[str, Any]:
        """
        Build a references section for the complete report based on Phase2 citation_index.json.

        This performs a light passthrough of all papers while intentionally
        dropping fields that are only meaningful within Phase2 ranking logic:
        - rank
        - rank_in_contribution
        - contribution_id
        - category
        """
        phase2_dir = base_dir / "phase2"
        citation_index_path = phase2_dir / "final" / "citation_index.json"
        references: Dict[str, Any] = {
            "source": str(citation_index_path),
            "items": [],
        }

        if not citation_index_path.exists():
            self.logger.warning(f"citation_index.json not found at: {citation_index_path}")
            return references

        try:
            data = json.loads(citation_index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            self.logger.warning(f"Failed to read citation_index.json: {e}")
            return references

        # Phase2 citation_index.json schema: prefer "items" (new), fallback to "papers" (legacy)
        papers = data.get("items", None)
        if papers is None:
            papers = data.get("papers") or []
        if not isinstance(papers, list):
            self.logger.warning("citation_index.json has unexpected structure: expected list under 'items' or 'papers'")
            return references

        drop_keys = {"rank", "rank_in_contribution", "contribution_id", "category"}
        cleaned_items = []
        for p in papers:
            if not isinstance(p, dict):
                continue
            cleaned = {k: v for k, v in p.items() if k not in drop_keys}
            # Ensure index is present and keep only well-formed entries
            if "index" in cleaned:
                cleaned_items.append(cleaned)

        # Sort by index for stable order
        cleaned_items.sort(key=lambda x: x.get("index", 0))
        references["items"] = cleaned_items
        return references
