"""
Phase 3 - Part 3: Unified Textual Similarity Detection

Execution: Runs as Part 3 of Phase 3 (after Part 1: Survey and Part 2: PDF Download)

Core Purpose: Unified textual similarity detection to avoid redundant LLM calls

Detection Scope:
- Core Task siblings (papers at the same taxonomy leaf node)
- All Contribution candidate papers

Key Features:
- Independent execution: Runs before Part 4/5 (Comparisons) to complete detection upfront
- Deduplication: Dedupe by canonical_id, each paper checked only once
- Result sharing: Part 4/5/6 all read from the same results file
- Text cache: Uses TextCache populated by Part 2 (no PDF downloads here)
- Skippable: Set SKIP_TEXTUAL_SIMILARITY=true to skip this part entirely

Output Files:
- phase3/textual_similarity_detection/results.json (detection results)
- phase3/textual_similarity_detection/metadata.json (metadata)
- phase3/textual_similarity_detection/per_paper/*.json (per-paper details)

Data Flow:
Part 2 populates TextCache → Part 3 generates results.json → Part 4/5 use results → Part 6 fills final report

Prerequisites:
- Part 2 (PDF Download) should run first to populate TextCache
- If TextCache is empty, this script will use abstracts as fallback

Usage:
  python -m scripts.run_phase3_textual_similarity \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1 \\
      --taxonomy output/paper123/phase3/core_task_survey/taxonomy.json
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from paper_novelty_pipeline.phases.phase3.textual_similarity_detector import TextualSimilarityDetector
from paper_novelty_pipeline.phases.phase3.llm_analyzer import LLMAnalyzer
from paper_novelty_pipeline.utils.text_cache import TextCache
from paper_novelty_pipeline.utils.phase3_io import (
    load_topk_core,
    load_contrib_file,
    read_paper_info,
)
from paper_novelty_pipeline.services.llm_client import create_llm_client


def setup_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logger for textual similarity detection."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"textual_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_phase2_data(
    phase2_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load Phase 2 output data.
    
    Args:
        phase2_dir: Phase 2 output directory
        logger: Logger instance
        
    Returns:
        Dict containing:
        - core_task_candidates: List of core task candidate papers
        - contribution_candidates_by_contrib: Dict[contrib_name, List[candidates]]
    """
    logger.info(f"Loading Phase 2 data from: {phase2_dir}")
    
    # Load Core Task candidates
    core_task_file = phase2_dir / "core_task_perfect_top50.json"
    if not core_task_file.exists():
        logger.warning(f"Core Task file not found: {core_task_file}")
        core_task_candidates = []
    else:
        core_task_candidates = load_topk_core(str(core_task_file))
        logger.info(f"Loaded {len(core_task_candidates)} Core Task candidates")
    
    # Load Contribution candidates
    # Directly glob for contribution files (simpler and more robust)
    contribution_candidates_by_contrib = {}
    contrib_files = sorted(phase2_dir.glob("contribution_*_perfect_top10.json"))
    
    if not contrib_files:
        logger.warning(f"No contribution files found in: {phase2_dir}")
    else:
        for contrib_file in contrib_files:
            # Extract contribution name from filename (e.g., "contribution_1" from "contribution_1_perfect_top10.json")
            contrib_name = contrib_file.stem.replace("_perfect_top10", "")
            
            try:
                candidates = load_contrib_file(str(contrib_file))
                contribution_candidates_by_contrib[contrib_name] = candidates
                logger.info(f"Loaded {len(candidates)} candidates for {contrib_name}")
            except Exception as e:
                logger.warning(f"Failed to load {contrib_file}: {e}")
    
    return {
        "core_task_candidates": core_task_candidates,
        "contribution_candidates_by_contrib": contribution_candidates_by_contrib
    }


def load_original_paper(
    phase1_dir: Path,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Load original paper metadata from Phase 1.
    
    Args:
        phase1_dir: Path to Phase 1 directory
        logger: Logger instance
        
    Returns:
        Paper info dict
    """
    logger.info(f"Loading original paper from: {phase1_dir}")
    
    if not phase1_dir.exists():
        logger.error(f"Phase1 directory not found: {phase1_dir}")
        raise FileNotFoundError(f"Phase1 directory not found: {phase1_dir}")
    
    paper_info = read_paper_info(str(phase1_dir))
    logger.info(f"Loaded original paper: {paper_info.get('title', 'Unknown')}")
    
    return paper_info


def load_taxonomy(
    taxonomy_path: Optional[Path],
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Load taxonomy data for relationship inference.
    
    Args:
        taxonomy_path: Path to taxonomy JSON
        logger: Logger instance
        
    Returns:
        Taxonomy dict or None if not available
    """
    if not taxonomy_path or not taxonomy_path.exists():
        logger.warning("Taxonomy file not provided or not found, sibling detection will be limited")
        return None
    
    logger.info(f"Loading taxonomy from: {taxonomy_path}")
    
    try:
        taxonomy = json.loads(taxonomy_path.read_text(encoding='utf-8'))
        logger.info("Taxonomy loaded successfully")
        return taxonomy
    except Exception as e:
        logger.error(f"Failed to load taxonomy: {e}")
        return None


def run_textual_similarity_detection(
    phase2_dir: Path,
    phase3_dir: Path,
    phase1_dir: Path,
    taxonomy_path: Optional[Path] = None,
    force_refresh_cache: bool = False,
    log_dir: Optional[Path] = None
) -> bool:
    """
    Run unified textual similarity detection for Phase 3 (Part 3).

    This function:
    1. Loads Phase 2 data and original paper
    2. Collects all unique papers to check (by canonical_id)
    3. Loads texts from TextCache (populated by Part 2: PDF Download)
    4. Runs TextualSimilarityDetector once per unique paper
    5. Saves results to phase3/textual_similarity_detection/

    Prerequisites:
    - Part 2 (PDF Download) should run first to populate TextCache
    - If TextCache is empty, papers will use abstracts as fallback

    Args:
        phase2_dir: Phase 2 output directory
        phase3_dir: Phase 3 output directory
        phase1_dir: Phase 1 directory containing original paper metadata
        taxonomy_path: Path to taxonomy JSON (optional, for sibling detection)
        force_refresh_cache: Force refresh text cache (not used in Part 3)
        log_dir: Directory for log files (optional)

    Returns:
        True if successful, False otherwise
    """
    # Setup logger
    logger = setup_logger("textual_similarity_detection", log_dir)

    logger.info("=" * 80)
    logger.info("Starting Part 3: Unified Textual Similarity Detection")
    logger.info("=" * 80)

    try:
        # 1. Initialize components
        logger.info("\n[Step 1] Initializing components...")

        text_cache_dir = phase3_dir / "cached_paper_texts"
        text_cache = TextCache(text_cache_dir, logger)
        logger.info(f"TextCache initialized at: {text_cache_dir}")

        # Check TextCache status and warn if empty
        cache_stats = text_cache.get_stats()
        if cache_stats["total_cached_papers"] == 0:
            logger.warning("⚠️  TextCache is empty! Part 2 (PDF Download) may not have run yet.")
            logger.warning("   Papers will use abstracts as fallback for similarity detection.")
            logger.warning("   For better results, run Part 2 first: python -m scripts.run_phase3_pdf_download ...")
        else:
            logger.info(f"   TextCache contains {cache_stats['total_cached_papers']} papers ({cache_stats['cache_size_mb']} MB)")

        llm_client = create_llm_client()
        logger.info("LLM client initialized (using default config)")

        llm_analyzer = LLMAnalyzer(
            llm_client=llm_client,
            output_base_dir=str(phase3_dir.parent),
            logger=logger
        )
        logger.info("LLMAnalyzer initialized")

        similarity_detector = TextualSimilarityDetector(llm_analyzer, logger)
        logger.info("TextualSimilarityDetector initialized")

        # 2. Load input data
        logger.info("\n[Step 2] Loading Phase 2 data...")
        phase2_data = load_phase2_data(phase2_dir, logger)

        logger.info("\n[Step 3] Loading original paper...")
        original_paper = load_original_paper(phase1_dir, logger)
        original_canonical_id = original_paper.get("canonical_id")

        logger.info("\n[Step 4] Loading taxonomy (optional)...")
        taxonomy = load_taxonomy(taxonomy_path, logger)

        # 3. Collect papers to check
        logger.info("\n[Step 5] Collecting papers to check...")
        papers_to_check = similarity_detector.collect_papers_to_check(
            core_task_candidates=phase2_data["core_task_candidates"],
            contribution_candidates_by_contrib=phase2_data["contribution_candidates_by_contrib"],
            taxonomy=taxonomy,
            original_paper_id=original_paper.get("canonical_id", "")
        )

        if not papers_to_check:
            logger.warning("No papers to check for textual similarity!")
            return False

        # 4. Get original paper full text (from Phase1 or TextCache)
        logger.info("\n[Step 6] Getting original paper full text...")

        # Try Phase1 fulltext file first (most reliable, no download needed)
        original_text = None
        phase1_fulltext = phase1_dir / "fulltext_cleaned.txt"
        if phase1_fulltext.exists():
            try:
                original_text = phase1_fulltext.read_text(encoding='utf-8')
                logger.info(f"✓ Loaded original paper text from Phase1 ({len(original_text)} chars)")
            except Exception as e:
                logger.warning(f"Failed to read Phase1 fulltext: {e}")
                original_text = None

        # Fallback: Try TextCache (populated by Part 2)
        if not original_text and original_canonical_id:
            original_text = text_cache.get_cached_text(original_canonical_id)
            if original_text:
                logger.info(f"✓ Original paper text loaded from TextCache ({len(original_text)} chars)")

        # Last resort: Use abstract
        if not original_text:
            original_text = original_paper.get("abstract", "")
            if original_text:
                logger.warning(f"⚠️  Using abstract as fallback for original paper ({len(original_text)} chars)")
            else:
                logger.error("Failed to get original paper full text!")
                return False

        logger.info(f"Original paper full text: {len(original_text)} chars")

        # 5. Load full texts from TextCache for all candidate papers
        logger.info("\n[Step 7] Loading full texts from TextCache (populated by Part 2)...")

        cache_hit_count = 0
        cache_miss_count = 0

        for paper in papers_to_check:
            canonical_id = paper["canonical_id"]

            # Check if already has full_text
            if paper.get("full_text"):
                cache_hit_count += 1
                continue

            # Try to load from TextCache
            cached_text = text_cache.get_cached_text(canonical_id)
            if cached_text:
                paper["full_text"] = cached_text
                cache_hit_count += 1
                logger.debug(f"  Cache hit: {canonical_id[:30]}... ({len(cached_text)} chars)")
            else:
                # Fallback to abstract if not in cache
                paper["full_text"] = paper.get("abstract", "")
                cache_miss_count += 1
                logger.debug(f"  Cache miss: {canonical_id[:30]}... (using abstract)")

        logger.info(f"  → {cache_hit_count} papers from TextCache, {cache_miss_count} papers using abstract fallback")

        if cache_miss_count > 0:
            logger.warning(f"⚠️  {cache_miss_count} papers not found in TextCache. Run Part 2 first for better results.")

        # 6. Run similarity detection
        logger.info("\n[Step 8] Running textual similarity detection...")
        output_base_dir = phase3_dir / "textual_similarity_detection"

        similarity_results = similarity_detector.detect_all_similarities(
            papers_to_check=papers_to_check,
            original_paper_text=original_text,
            output_base_dir=output_base_dir
        )

        # 7. Save results
        logger.info("\n[Step 9] Saving results...")
        output_dir = phase3_dir / "textual_similarity_detection"
        output_dir.mkdir(parents=True, exist_ok=True)

        # 7.1 Main results file
        aggregated_results = similarity_detector.get_aggregated_results()

        results_file = output_dir / "results.json"
        results_file.write_text(
            json.dumps(aggregated_results, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        logger.info(f"✓ Results saved to: {results_file}")

        # 7.2 Metadata file
        final_cache_stats = text_cache.get_stats()
        metadata = {
            "detection_timestamp": datetime.now().isoformat(),
            "phase2_dir": str(phase2_dir),
            "phase3_dir": str(phase3_dir),
            "original_paper": {
                "canonical_id": original_canonical_id,
                "title": original_paper.get("title")
            },
            "total_papers_checked": len(papers_to_check),
            "papers_with_similarities": len(aggregated_results["papers_with_plagiarism"]),
            "papers_without_similarities": len(aggregated_results["papers_without_plagiarism"]),
            "total_unique_segments": aggregated_results["statistics"]["total_unique_segments"],
            "cache_stats": final_cache_stats,
            "text_loading_stats": {
                "cache_hits": cache_hit_count,
                "cache_misses": cache_miss_count
            },
            "detection_config": {
                "force_refresh_cache": force_refresh_cache,
                "taxonomy_used": taxonomy is not None
            }
        }
        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        logger.info(f"✓ Metadata saved to: {metadata_file}")

        # 8. Summary
        logger.info("\n" + "=" * 80)
        logger.info("PART 3: DETECTION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Papers checked: {len(papers_to_check)}")
        logger.info(f"Papers with similarities: {metadata['papers_with_similarities']}")
        logger.info(f"Total unique segments: {metadata['total_unique_segments']}")
        logger.info(f"TextCache stats: {final_cache_stats['total_cached_papers']} papers ({final_cache_stats['cache_size_mb']} MB)")
        logger.info(f"Results directory: {output_dir}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"Textual similarity detection failed: {e}", exc_info=True)
        return False


def main():
    """CLI entry point for textual similarity detection (Part 3)."""
    # Early skip check - if SKIP_TEXTUAL_SIMILARITY is set, skip entire script
    if os.getenv("SKIP_TEXTUAL_SIMILARITY", "false").lower() == "true":
        print("=" * 70)
        print("Part 3: Textual Similarity Detection - SKIPPED")
        print("=" * 70)
        print("SKIP_TEXTUAL_SIMILARITY=true is set in environment.")
        print("Skipping Part 3 entirely. Part 4/5/6 will proceed without similarity data.")
        print("=" * 70)
        return 0

    parser = argparse.ArgumentParser(
        description="Part 3: Run unified textual similarity detection for Phase 3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Prerequisites:
  Part 2 (PDF Download) should run first to populate TextCache.
  If TextCache is empty, papers will use abstracts as fallback.

Environment Variables:
  SKIP_TEXTUAL_SIMILARITY=true  Skip this entire script

Examples:
  # Run with taxonomy (recommended)
  python -m scripts.run_phase3_textual_similarity \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1 \\
      --taxonomy output/paper123/phase3/core_task_survey/taxonomy.json

  # Run without taxonomy
  python -m scripts.run_phase3_textual_similarity \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1
"""
    )

    parser.add_argument(
        "--phase2-dir",
        type=Path,
        required=True,
        help="Phase 2 output directory (e.g., output/paper123/phase2/final)"
    )
    parser.add_argument(
        "--phase3-dir",
        type=Path,
        required=True,
        help="Phase 3 output directory (e.g., output/paper123/phase3)"
    )
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        required=True,
        help="Phase 1 directory containing original paper metadata (e.g., output/paper123/phase1)"
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help="Taxonomy JSON file (optional, for sibling detection)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh text cache (not typically needed for Part 3)"
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Directory for log files (optional)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.phase2_dir.exists():
        print(f"ERROR: Phase 2 directory not found: {args.phase2_dir}")
        return 1

    if not args.phase1_dir.exists():
        print(f"ERROR: Phase 1 directory not found: {args.phase1_dir}")
        return 1

    # Create phase3_dir if it doesn't exist
    args.phase3_dir.mkdir(parents=True, exist_ok=True)

    # Run detection
    success = run_textual_similarity_detection(
        phase2_dir=args.phase2_dir,
        phase3_dir=args.phase3_dir,
        phase1_dir=args.phase1_dir,
        taxonomy_path=args.taxonomy,
        force_refresh_cache=args.force_refresh,
        log_dir=args.log_dir
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

