"""
Unified textual similarity detection for Phase 3.

This module ensures each paper (by canonical_id) is only analyzed once for
textual similarity, avoiding redundant LLM calls and ensuring consistency.

Detection criteria:
1. Core Task siblings (same taxonomy leaf as original paper)
2. All Contribution candidates

"""

from typing import Dict, List, Optional, Any
import logging
import dataclasses
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from paper_novelty_pipeline.config import PHASE3_MIN_SIMILARITY_WORDS

# Similarity validation threshold (must match llm_analyzer.py)
MIN_SIMILARITY_WORDS = PHASE3_MIN_SIMILARITY_WORDS


class TextualSimilarityDetector:
    """
    Centralized textual similarity detection for Phase 3.
    
    This class handles:
    - Collecting all papers that need similarity detection
    - Deduplicating by canonical_id
    - Performing batch detection
    - Caching results for reuse
    - Aggregating results for the complete report
    
    Usage:
        detector = TextualSimilarityDetector(llm_analyzer, logger)
        papers = detector.collect_papers_to_check(...)
        detector.detect_all_similarities(papers, original_text, output_dir)
        segments = detector.get_segments_for_paper(canonical_id)
    """
    
    def __init__(self, llm_analyzer, logger: logging.Logger):
        """
        Initialize the detector.
        
        Args:
            llm_analyzer: LLMAnalyzer instance for performing detection
            logger: Logger instance for logging
        """
        self.llm_analyzer = llm_analyzer
        self.logger = logger
        self.similarity_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = threading.Lock()  # Thread-safe cache operations
    
    def collect_papers_to_check(
        self,
        core_task_candidates: List[Dict],
        contribution_candidates_by_contrib: Dict[str, List[Any]],
        taxonomy: Optional[Dict],
        original_paper_id: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Collect all unique papers (by canonical_id) that need similarity detection.
        
        Detection criteria (any of):
        1. Core Task sibling (same taxonomy leaf as original paper)
        2. Any Contribution candidate
        
        Args:
            core_task_candidates: List of core task candidate papers
            contribution_candidates_by_contrib: Dict of {contribution_name: [candidates]}
            taxonomy: Taxonomy structure for relationship inference
            original_paper_id: Original paper's ID for taxonomy lookup
            
        Returns:
            List of unique papers with metadata:
            [
                {
                    "canonical_id": str,
                    "paper_id": str,
                    "title": str,
                    "abstract": str,
                    "full_text": str,
                    "sources": List[str],
                    "is_core_task_sibling": bool,
                    "contribution_sources": List[str]
                }
            ]
        """
        from paper_novelty_pipeline.utils.phase3_io import canonical_id_from_paper_info
        
        papers_map: Dict[str, Dict[str, Any]] = {}
        
        # Helper function to infer relationship from taxonomy
        def infer_relationship(candidate: Dict) -> str:
            """Infer whether candidate is sibling or non-sibling."""
            if not taxonomy or not original_paper_id:
                return "unknown"
            
            # Get candidate's canonical_id or paper_id
            cand_id = candidate.get("canonical_id") or candidate.get("paper_id")
            if not cand_id:
                return "unknown"
            
            # Try mapping-based approach first (for backward compatibility)
            mapping = taxonomy.get("mapping", [])
            if mapping:
                # Find taxonomy paths for both papers
                orig_path = None
                cand_path = None
                
                for item in mapping:
                    item_id = item.get("paper_id")
                    if item_id == original_paper_id:
                        orig_path = item.get("taxonomy_path", [])
                    if item_id == cand_id:
                        cand_path = item.get("taxonomy_path", [])
                
                # Compare paths - siblings have identical paths
                if orig_path and cand_path and orig_path == cand_path:
                    return "sibling"
                return "non_sibling"
            
            # Fallback: Tree-based approach (for tree-structured taxonomy)
            # Find which leaf categories contain each paper
            def find_paper_in_tree(paper_id, node, path=[]):
                """Recursively find paper in taxonomy tree, return leaf path."""
                # Check if current node has papers list
                if "papers" in node and paper_id in node["papers"]:
                    return path + [node.get("name", "")]
                
                # Recurse into subtopics
                if "subtopics" in node:
                    for subtopic in node["subtopics"]:
                        result = find_paper_in_tree(paper_id, subtopic, path + [node.get("name", "")])
                        if result:
                            return result
                return None
            
            orig_path = find_paper_in_tree(original_paper_id, taxonomy)
            cand_path = find_paper_in_tree(cand_id, taxonomy)
            
            # Siblings have identical leaf paths
            if orig_path and cand_path and orig_path == cand_path:
                return "sibling"
            elif orig_path and cand_path:
                return "non_sibling"
            return "unknown"
        
        # ===== 1. Collect Core Task siblings =====
        self.logger.info("Collecting Core Task candidates for similarity detection...")
        
        for candidate in core_task_candidates:
            canonical_id = canonical_id_from_paper_info(candidate)
            if not canonical_id:
                continue
            
            relationship = infer_relationship(candidate)
            
            # Only check siblings for Core Task
            if relationship == "sibling":
                if canonical_id not in papers_map:
                    papers_map[canonical_id] = {
                        "canonical_id": canonical_id,
                        "paper_id": candidate.get("paper_id", ""),
                        "title": candidate.get("title", ""),
                        "abstract": candidate.get("abstract", ""),
                        "full_text": candidate.get("full_text", ""),
                        "pdf_url": candidate.get("pdf_url"),
                        "source_url": candidate.get("source_url"),
                        "doi": candidate.get("doi"),
                        "arxiv_id": candidate.get("arxiv_id"),
                        "year": candidate.get("year", 0),
                        "venue": candidate.get("venue", ""),
                        "authors": candidate.get("authors", []),
                        "sources": ["Core Task (sibling)"],
                        "is_core_task_sibling": True,
                        "contribution_sources": []
                    }
                else:
                    # Already exists (from contribution), mark as core task sibling
                    if "Core Task (sibling)" not in papers_map[canonical_id]["sources"]:
                        papers_map[canonical_id]["sources"].append("Core Task (sibling)")
                    papers_map[canonical_id]["is_core_task_sibling"] = True
        
        # ===== 2. Collect all Contribution candidates =====
        self.logger.info("Collecting Contribution candidates for similarity detection...")
        
        for contrib_name, candidates in contribution_candidates_by_contrib.items():
            for candidate in candidates:
                canonical_id = canonical_id_from_paper_info(candidate)
                
                if not canonical_id:
                    self.logger.warning(
                        f"Skipping candidate without canonical_id: "
                        f"{candidate.get('title', 'Unknown')}"
                    )
                    continue
                
                if canonical_id not in papers_map:
                    # New paper
                    papers_map[canonical_id] = {
                        "canonical_id": canonical_id,
                        "paper_id": candidate.get('paper_id', ''),
                        "title": candidate.get('title', ''),
                        "abstract": candidate.get('abstract', '') or '',
                        "full_text": candidate.get('full_text', '') or '',
                        "pdf_url": candidate.get('pdf_url'),
                        "source_url": candidate.get('source_url'),
                        "doi": candidate.get('doi'),
                        "arxiv_id": candidate.get('arxiv_id'),
                        "year": candidate.get('year', 0),
                        "venue": candidate.get('venue', ''),
                        "authors": candidate.get('authors', []),
                        "sources": [f"Contribution: {contrib_name}"],
                        "is_core_task_sibling": False,
                        "contribution_sources": [contrib_name]
                    }
                else:
                    # Already exists (possibly from Core Task or another contribution)
                    source = f"Contribution: {contrib_name}"
                    if source not in papers_map[canonical_id]["sources"]:
                        papers_map[canonical_id]["sources"].append(source)
                    if contrib_name not in papers_map[canonical_id]["contribution_sources"]:
                        papers_map[canonical_id]["contribution_sources"].append(contrib_name)
        
        papers_list = list(papers_map.values())
        
        # Log statistics
        core_task_count = sum(p["is_core_task_sibling"] for p in papers_list)
        contrib_count = sum(len(p["contribution_sources"]) > 0 for p in papers_list)
        overlap_count = sum(
            p["is_core_task_sibling"] and len(p["contribution_sources"]) > 0 
            for p in papers_list
        )
        
        self.logger.info(
            f"Collected {len(papers_list)} unique papers for similarity detection:\n"
            f"  - Core Task siblings: {core_task_count}\n"
            f"  - Contribution candidates: {contrib_count}\n"
            f"  - Overlap (both): {overlap_count}"
        )
        
        return papers_list
    
    def detect_all_similarities(
        self,
        papers_to_check: List[Dict],
        original_paper_text: str,
        output_base_dir: Path,
        max_workers: int = 3
    ) -> Dict[str, Dict[str, Any]]:
        """
        Detect textual similarities for all papers in parallel.
        
        This method calls the LLM once per paper to detect textual similarities,
        then caches the results for reuse by Core Task and Contribution analyses.
        
        Args:
            papers_to_check: List of papers to check (from collect_papers_to_check)
            original_paper_text: Full text of the original paper
            output_base_dir: Base directory for saving raw LLM responses
            max_workers: Maximum concurrent LLM calls (default: 2)
            
        Returns:
            Cache dictionary: {canonical_id: {segments: [...], sources: [...], paper_info: {...}}}
        """
        if not papers_to_check:
            self.logger.info("No papers to check for textual similarity")
            return self.similarity_cache
        
        self.logger.info(
            f"Starting parallel similarity detection for {len(papers_to_check)} papers "
            f"(max_workers={max_workers})"
        )
        
        def process_one_paper(paper: Dict) -> tuple:
            """Process a single paper and return (canonical_id, result_dict)."""
            canonical_id = paper["canonical_id"]
            try:
                segments = self._detect_similarity_for_paper(
                    original_text=original_paper_text,
                    candidate_text=paper.get("full_text") or paper.get("abstract", ""),
                    candidate_info=paper,
                    output_base_dir=output_base_dir
                )
                
                result = {
                    "segments": segments,
                    "sources": paper["sources"],
                    "paper_info": {
                        "canonical_id": canonical_id,
                        "paper_id": paper["paper_id"],
                        "title": paper["title"]
                    }
                }
                
                # Save per-paper evaluation file
                if output_base_dir:
                    self._save_per_paper_result(
                        paper=paper,
                        segments=segments,
                        output_base_dir=output_base_dir
                    )
                
                return (canonical_id, result, len(segments), None)
            
            except Exception as e:
                # Return error result
                result = {
                    "segments": [],
                    "sources": paper["sources"],
                    "paper_info": {
                        "canonical_id": canonical_id,
                        "paper_id": paper["paper_id"],
                        "title": paper["title"]
                    },
                    "error": str(e)
                }
                
                # Save per-paper evaluation file (error case)
                if output_base_dir:
                    self._save_per_paper_result(
                        paper=paper,
                        segments=[],
                        output_base_dir=output_base_dir,
                        error=str(e)
                    )
                
                return (canonical_id, result, 0, str(e))
        
        # Execute in parallel
        completed = 0
        success_count = 0
        error_count = 0
        total_segments = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_one_paper, paper): paper for paper in papers_to_check}
            
            for future in as_completed(futures):
                paper = futures[future]
                completed += 1
                
                try:
                    canonical_id, result, seg_count, error = future.result()
                    
                    # Thread-safe cache update
                    with self._cache_lock:
                        self.similarity_cache[canonical_id] = result
                    
                    if error:
                        error_count += 1
                        self.logger.warning(
                            f"[{completed}/{len(papers_to_check)}] {canonical_id[:20]}... -> error: {error}"
                        )
                    else:
                        success_count += 1
                        total_segments += seg_count
                        self.logger.info(
                            f"[{completed}/{len(papers_to_check)}] {canonical_id[:20]}... -> {seg_count} segments"
                        )
                
                except Exception as e:
                    error_count += 1
                    canonical_id = paper.get("canonical_id", "unknown")
                    self.logger.error(
                        f"[{completed}/{len(papers_to_check)}] {canonical_id[:20]}... -> unexpected error: {e}"
                    )
                    
                    # Thread-safe cache update for unexpected errors
                    with self._cache_lock:
                        self.similarity_cache[canonical_id] = {
                            "segments": [],
                            "sources": paper.get("sources", []),
                            "paper_info": {
                                "canonical_id": canonical_id,
                                "paper_id": paper.get("paper_id", ""),
                                "title": paper.get("title", "")
                            },
                            "error": str(e)
                        }
        
        self.logger.info(
            f"Completed parallel similarity detection: "
            f"{success_count} success, {error_count} errors, {total_segments} total segments. "
            f"Cache contains {len(self.similarity_cache)} entries."
        )
        
        return self.similarity_cache
    
    def get_segments_for_paper(self, canonical_id: Optional[str]) -> List[Dict]:
        """
        Retrieve cached similarity segments for a paper.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            List of similarity segment dicts (empty list if not found)
        """
        if not canonical_id:
            return []
        
        if canonical_id in self.similarity_cache:
            cache_entry = self.similarity_cache[canonical_id]
            return cache_entry.get("segments", [])
        
        return []
    
    def _detect_similarity_for_paper(
        self,
        original_text: str,
        candidate_text: str,
        candidate_info: Dict[str, Any],
        output_base_dir: Path
    ) -> List[Dict]:
        """
        Call LLM to detect textual similarities for a single paper.
        
        This method reuses existing llm_analyzer methods for consistency.
        
        Args:
            original_text: Full text of the original paper
            candidate_text: Full text (or abstract) of the candidate paper
            candidate_info: Candidate paper metadata
            output_base_dir: Base directory for saving raw responses
            
        Returns:
            List of similarity segment dicts
        """
        # Normalize texts for verification
        orig_norm = self.llm_analyzer.evidence_verifier.normalize_text(original_text)
        cand_norm = self.llm_analyzer.evidence_verifier.normalize_text(candidate_text)
        
        if not orig_norm or not cand_norm:
            self.logger.warning(
                f"Empty normalized text for {candidate_info['canonical_id']}, skipping"
            )
            return []
        
        # Create a temporary RetrievedPaper-like object for compatibility with existing methods
        from paper_novelty_pipeline.models import RetrievedPaper
        
        candidate_paper = RetrievedPaper(
            paper_id=candidate_info["paper_id"],
            title=candidate_info["title"],
            abstract=candidate_info["abstract"],
            authors=[],
            year=0,
            venue="",
            relevance_score=0.0,
            source_url=None
        )
        
        # Build unified prompt with XML-wrapped data (single user message)
        complete_prompt = self.llm_analyzer._build_unified_similarity_prompt(
            orig_text_normalized=orig_norm,
            cand_text_normalized=cand_norm,
            candidate_title=candidate_paper.title
        )
        
        messages = [
            {"role": "user", "content": complete_prompt}
        ]
        
        # Call LLM for Task 2: Unified Similarity Detection
        # Network retries are handled by the LLM client itself (MAX_RETRIES).
        from paper_novelty_pipeline.phases.phase3.utils import parse_json_flexible
        
        similarity_segments = []
        
        try:
            self.logger.info(f"LLM similarity detection for {candidate_info['canonical_id']}")
            
            # Call LLM using generate() instead of generate_json()
            # This allows LLM to return either dict or list format
            result_text = self.llm_analyzer.llm_client.generate(
                messages,
                max_tokens=8000
            )
            
            if not result_text:
                self.logger.warning(f"LLM returned empty response for {candidate_info['canonical_id']}")
                return []
            
            # Parse the text response manually
            result = parse_json_flexible(result_text)
            
            if result:
                # Save raw response
                self.llm_analyzer._save_llm_response(
                    result,
                    "unified_similarity",
                    candidate_info["canonical_id"],
                    context="unified_detection"
                )
                
                # Handle both dict format (with "similarity_segments" key) and direct list format
                if isinstance(result, list):
                    # LLM returned a direct list instead of {"similarity_segments": [...]}
                    self.logger.debug(f"LLM returned direct list format for {candidate_info['canonical_id']}, wrapping...")
                    result = {"similarity_segments": result}
                
                if isinstance(result, dict):
                    # Use existing _parse_similarity_segments method
                    temp_segments = self.llm_analyzer._parse_similarity_segments(
                        result,
                        candidate=candidate_paper,
                        orig_text_norm=orig_norm,
                        cand_text_norm=cand_norm
                    )
                    
                    # Filter segments: only keep those that are verified AND meet word count
                    similarity_segments = [
                        seg for seg in temp_segments
                        if (seg.original_location.found and 
                            seg.candidate_location.found and
                            seg.word_count >= MIN_SIMILARITY_WORDS)
                    ]
                    
                    # Log filtering result
                    if len(temp_segments) > len(similarity_segments):
                        self.logger.info(
                            f"Filtered {len(temp_segments)} → {len(similarity_segments)} segments "
                            f"(verified & word_count >= {MIN_SIMILARITY_WORDS}) "
                            f"for {candidate_info['canonical_id']}"
                        )
            else:
                self.logger.warning(f"Failed to parse similarity response for {candidate_info['canonical_id']}")
        
        except Exception as e:
            self.logger.error(
                f"Similarity detection failed for {candidate_info['canonical_id']}: {e}",
                exc_info=True
            )
            return []
        
        # Convert TextualSimilaritySegment objects to dicts (using plagiarism format)
        return [seg.to_dict() for seg in similarity_segments]
    
    def get_aggregated_results(self) -> Dict[str, Any]:
        """
        Generate aggregated plagiarism detection results for saving to results.json.
        
        Returns:
            Dictionary with papers_with_plagiarism and papers_without_plagiarism
        """
        from datetime import datetime
        
        papers_with_plagiarism = []
        papers_without_plagiarism = []
        total_segments = 0
        
        for canonical_id, cache_entry in self.similarity_cache.items():
            paper_info = cache_entry.get("paper_info", {})
            segments = cache_entry.get("segments", [])
            sources = cache_entry.get("sources", [])
            error = cache_entry.get("error")
            
            if segments:
                # Has plagiarism
                papers_with_plagiarism.append({
                    "paper_id": paper_info.get("paper_id", canonical_id),
                    "canonical_id": canonical_id,
                    "title": paper_info.get("title", ""),
                    "sources": sources,
                    "plagiarism_segments": segments
                })
                total_segments += len(segments)
            else:
                # No plagiarism
                reason = "error" if error else "no_plagiarism_detected"
                papers_without_plagiarism.append({
                    "paper_id": paper_info.get("paper_id", canonical_id),
                    "canonical_id": canonical_id,
                    "title": paper_info.get("title", ""),
                    "sources": sources,
                    "checked_at": datetime.now().isoformat(),
                    "reason": reason
                })
                if error:
                    papers_without_plagiarism[-1]["error_message"] = error
        
        # Sort papers_with_plagiarism by number of segments (most plagiarism first)
        papers_with_plagiarism.sort(key=lambda p: len(p["plagiarism_segments"]), reverse=True)
        
        return {
            "papers_with_plagiarism": papers_with_plagiarism,
            "papers_without_plagiarism": papers_without_plagiarism,
            "statistics": {
                "total_papers_checked": len(self.similarity_cache),
                "papers_with_plagiarism": len(papers_with_plagiarism),
                "papers_without_plagiarism": len(papers_without_plagiarism),
                "total_unique_segments": total_segments
            }
        }
    
    def _save_per_paper_result(
        self, 
        paper: Dict[str, Any], 
        segments: List[Dict], 
        output_base_dir: Path,
        error: Optional[str] = None
    ) -> None:
        """
        Save per-paper evaluation result to dedicated file.
        
        Args:
            paper: Paper metadata with canonical_id, title, sources, etc.
            segments: List of similarity segments (may be empty)
            output_base_dir: Base directory for output
            error: Optional error message if detection failed
        """
        import json
        from datetime import datetime
        
        try:
            per_paper_dir = Path(output_base_dir) / "per_paper"
            per_paper_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate safe filename from canonical_id
            canonical_id = paper.get("canonical_id", "unknown")
            safe_filename = self._sanitize_filename(canonical_id) + ".json"
            output_file = per_paper_dir / safe_filename
            
            # Build per-paper result structure
            per_paper_result = {
                "paper_info": {
                    "canonical_id": canonical_id,
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "authors": paper.get("authors", []),
                    "year": paper.get("year"),
                    "venue": paper.get("venue", "")
                },
                "detection_metadata": {
                    "checked_at": datetime.now().isoformat(),
                    "text_source": "pdf_full" if len(paper.get("full_text", "")) > 1000 else "abstract",
                    "text_length": len(paper.get("full_text", "") or paper.get("abstract", "")),
                    "sources": paper.get("sources", [])
                },
                "similarity_result": {
                    "has_similarities": len(segments) > 0,
                    "total_segments": len(segments),
                    "segments": segments
                }
            }
            
            if error:
                per_paper_result["detection_metadata"]["error"] = error
                per_paper_result["similarity_result"]["error_occurred"] = True
            
            # Write to file
            output_file.write_text(
                json.dumps(per_paper_result, indent=2, ensure_ascii=False),
                encoding='utf-8'
            )
            
            self.logger.debug(f"Saved per-paper result to: {output_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save per-paper result for {paper.get('canonical_id')}: {e}")
    
    def _sanitize_filename(self, text: str) -> str:
        """
        Convert canonical_id or any text to filesystem-safe filename.
        
        Examples:
            arxiv:2403.01460 → arxiv_2403.01460
            doi:10.48550/arxiv.2403.01460 → doi_10.48550_arxiv.2403.01460
            openreview:KAGR7Mqu4h → openreview_KAGR7Mqu4h
        """
        import re
        # Replace unsafe characters with underscore
        safe_text = re.sub(r'[^\w\-.]', '_', text or "unknown")
        # Limit length
        return safe_text[:150] if len(safe_text) > 150 else safe_text

