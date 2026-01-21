"""
Phase 3 - Part 2: PDF Download & Text Extraction

This script downloads PDFs and extracts text for all candidate papers,
populating the TextCache for use by subsequent Phase 3 components.

Execution Order: After Part 1 (Survey), before Part 3 (Similarity Detection)

Core Functions:
- Load Phase 2 data (core_task + contribution candidates)
- Load original paper full text from phase1/fulltext_cleaned.txt
- Parallel download all candidate PDFs
- Extract text and cache to TextCache
- Cleanup temporary PDFs

Output:
- phase3/cached_paper_texts/*.json (cached paper texts)
- phase3/pdf_extraction/metadata.json (extraction statistics)

Usage:
    python -m scripts.run_phase3_pdf_download \\
        --phase2-dir output/paper123/phase2/final \\
        --phase3-dir output/paper123/phase3 \\
        --phase1-dir output/paper123/phase1 \\
        --max-workers 3
"""

import json
import logging
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from paper_novelty_pipeline.utils.text_cache import TextCache
from paper_novelty_pipeline.phases.phase3.pdf_handler import PDFHandler
from paper_novelty_pipeline.utils.phase3_io import (
    load_topk_core,
    load_contrib_file,
    read_paper_info,
    canonical_id_from_paper_info,
)
from paper_novelty_pipeline.config import PDF_DOWNLOAD_DIR
from paper_novelty_pipeline.models import RetrievedPaper


def cleanup_temp_pdfs(logger: logging.Logger) -> int:
    """
    Clean up all downloaded PDF files in temp_pdfs/.

    Args:
        logger: Logger instance

    Returns:
        Number of PDFs deleted
    """
    pdf_dir = Path(PDF_DOWNLOAD_DIR)
    if not pdf_dir.exists():
        logger.info("No PDF directory to clean up")
        return 0

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        logger.info("No PDF files to clean up")
        return 0

    logger.info(f"Found {len(pdf_files)} PDF files to delete...")
    deleted_count = 0
    failed_count = 0

    for pdf_file in pdf_files:
        try:
            pdf_file.unlink()
            deleted_count += 1
            logger.debug(f"  Deleted: {pdf_file.name}")
        except Exception as e:
            failed_count += 1
            logger.warning(f"  Failed to delete {pdf_file.name}: {e}")

    if failed_count > 0:
        logger.warning(f"PDF cleanup completed with errors: {deleted_count} deleted, {failed_count} failed")
    else:
        logger.info(f"PDF cleanup completed: {deleted_count} files deleted")

    return deleted_count


def setup_logger(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Setup logger for PDF download."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

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
            log_dir / f"pdf_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    contribution_candidates_by_contrib = {}
    contrib_files = sorted(phase2_dir.glob("contribution_*_perfect_top10.json"))

    if not contrib_files:
        logger.warning(f"No contribution files found in: {phase2_dir}")
    else:
        for contrib_file in contrib_files:
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


def collect_all_papers(
    phase2_data: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Collect all unique papers (by canonical_id) that need PDF download.

    Args:
        phase2_data: Phase 2 data dict
        logger: Logger instance

    Returns:
        List of unique papers with metadata
    """
    papers_map: Dict[str, Dict[str, Any]] = {}

    # Collect Core Task candidates
    logger.info("Collecting Core Task candidates...")
    for candidate in phase2_data["core_task_candidates"]:
        canonical_id = canonical_id_from_paper_info(candidate)
        if not canonical_id:
            continue

        if canonical_id not in papers_map:
            papers_map[canonical_id] = {
                "canonical_id": canonical_id,
                "paper_id": candidate.get("paper_id", ""),
                "title": candidate.get("title", ""),
                "abstract": candidate.get("abstract", ""),
                "pdf_url": candidate.get("pdf_url"),
                "source_url": candidate.get("source_url"),
                "doi": candidate.get("doi"),
                "arxiv_id": candidate.get("arxiv_id"),
                "year": candidate.get("year", 0),
                "venue": candidate.get("venue", ""),
                "authors": candidate.get("authors", []),
                "sources": ["core_task"],
            }
        else:
            if "core_task" not in papers_map[canonical_id]["sources"]:
                papers_map[canonical_id]["sources"].append("core_task")

    # Collect Contribution candidates
    logger.info("Collecting Contribution candidates...")
    for contrib_name, candidates in phase2_data["contribution_candidates_by_contrib"].items():
        for candidate in candidates:
            canonical_id = canonical_id_from_paper_info(candidate)
            if not canonical_id:
                continue

            if canonical_id not in papers_map:
                papers_map[canonical_id] = {
                    "canonical_id": canonical_id,
                    "paper_id": candidate.get("paper_id", ""),
                    "title": candidate.get("title", ""),
                    "abstract": candidate.get("abstract", ""),
                    "pdf_url": candidate.get("pdf_url"),
                    "source_url": candidate.get("source_url"),
                    "doi": candidate.get("doi"),
                    "arxiv_id": candidate.get("arxiv_id"),
                    "year": candidate.get("year", 0),
                    "venue": candidate.get("venue", ""),
                    "authors": candidate.get("authors", []),
                    "sources": [contrib_name],
                }
            else:
                if contrib_name not in papers_map[canonical_id]["sources"]:
                    papers_map[canonical_id]["sources"].append(contrib_name)

    papers_list = list(papers_map.values())
    logger.info(f"Collected {len(papers_list)} unique papers for PDF download")

    return papers_list


def run_pdf_download(
    phase2_dir: Path,
    phase3_dir: Path,
    phase1_dir: Path,
    max_workers: int = 3,
    force_refresh: bool = False,
    log_dir: Optional[Path] = None
) -> bool:
    """
    Run PDF download and text extraction for Phase 3.

    This function:
    1. Loads Phase 2 data and original paper
    2. Collects all unique papers to download (by canonical_id)
    3. Downloads PDFs and extracts text in parallel
    4. Caches extracted texts to TextCache
    5. Cleans up temporary PDFs

    Args:
        phase2_dir: Phase 2 output directory
        phase3_dir: Phase 3 output directory
        phase1_dir: Phase 1 directory containing original paper metadata
        max_workers: Maximum parallel downloads (default: 3)
        force_refresh: Force refresh text cache
        log_dir: Directory for log files (optional)

    Returns:
        True if successful, False otherwise
    """
    logger = setup_logger("pdf_download", log_dir)

    logger.info("=" * 80)
    logger.info("Phase 3 - Part 2: PDF Download & Text Extraction")
    logger.info("=" * 80)

    try:
        # Step 1: Initialize components
        logger.info("\n[Step 1] Initializing components...")

        text_cache_dir = phase3_dir / "cached_paper_texts"
        text_cache = TextCache(text_cache_dir, logger)
        logger.info(f"TextCache initialized at: {text_cache_dir}")

        pdf_handler = PDFHandler(
            output_base_dir=str(phase3_dir.parent),
            logger=logger,
            text_cache=text_cache
        )
        logger.info("PDFHandler initialized with TextCache integration")

        # Step 2: Load Phase 2 data
        logger.info("\n[Step 2] Loading Phase 2 data...")
        phase2_data = load_phase2_data(phase2_dir, logger)

        # Step 3: Load original paper
        logger.info("\n[Step 3] Loading original paper...")
        original_paper = read_paper_info(str(phase1_dir))
        original_canonical_id = original_paper.get("canonical_id", "")
        logger.info(f"Original paper: {original_paper.get('title', 'Unknown')}")

        # Step 4: Cache original paper text
        logger.info("\n[Step 4] Caching original paper text...")
        phase1_fulltext = phase1_dir / "fulltext_cleaned.txt"
        if phase1_fulltext.exists():
            try:
                original_text = phase1_fulltext.read_text(encoding='utf-8')
                logger.info(f"Loaded original paper text from Phase1 ({len(original_text)} chars)")

                if original_canonical_id:
                    text_cache.cache_text(original_canonical_id, original_text, {
                        "source": "phase1_fulltext",
                        "canonical_id": original_canonical_id,
                        "title": original_paper.get("title", ""),
                    })
                    logger.info(f"Cached original paper text")
            except Exception as e:
                logger.warning(f"Failed to read Phase1 fulltext: {e}")
        else:
            logger.warning(f"Phase1 fulltext not found: {phase1_fulltext}")

        # Step 5: Collect all papers to download
        logger.info("\n[Step 5] Collecting papers to download...")
        papers_to_download = collect_all_papers(phase2_data, logger)

        if not papers_to_download:
            logger.warning("No papers to download!")
            return False

        # Step 6: Check cache and prepare download list
        logger.info("\n[Step 6] Checking cache and preparing download list...")
        papers_to_fetch = []
        cached_count = 0

        for paper in papers_to_download:
            canonical_id = paper["canonical_id"]

            # Check cache (unless force refresh)
            if not force_refresh:
                cached_text = text_cache.get_cached_text(canonical_id)
                if cached_text:
                    cached_count += 1
                    continue

            # Need to fetch - create RetrievedPaper object
            retrieved_paper = RetrievedPaper(
                paper_id=canonical_id,
                title=paper.get("title", ""),
                abstract=paper.get("abstract", ""),
                authors=paper.get("authors", []),
                venue=paper.get("venue", ""),
                year=paper.get("year", 0),
                doi=paper.get("doi"),
                arxiv_id=paper.get("arxiv_id"),
                source_url=paper.get("source_url"),
                pdf_url=paper.get("pdf_url"),
                canonical_id=canonical_id
            )
            papers_to_fetch.append(retrieved_paper)

        logger.info(f"  Cache hits: {cached_count}")
        logger.info(f"  Papers to fetch: {len(papers_to_fetch)}")

        # Step 7: Parallel download
        if papers_to_fetch:
            logger.info(f"\n[Step 7] Downloading PDFs (max_workers={max_workers})...")
            download_results = pdf_handler.download_batch(papers_to_fetch, max_workers=max_workers)

            # Count results
            success_count = sum(1 for _, src, _ in download_results.values() if src in ("pdf_full", "fulltext", "constructed_pdf"))
            cached_during_download = sum(1 for _, src, _ in download_results.values() if src == "cached")
            fallback_count = sum(1 for _, src, _ in download_results.values() if src == "abstract" or src == "abstract_fallback")
            error_count = sum(1 for _, src, _ in download_results.values() if src == "error")

            logger.info(f"Download results: {success_count} PDF, {cached_during_download} cached, {fallback_count} abstract fallback, {error_count} errors")
        else:
            logger.info("\n[Step 7] All papers already cached, skipping download")
            download_results = {}

        # Step 8: Save metadata
        logger.info("\n[Step 8] Saving metadata...")
        output_dir = phase3_dir / "pdf_extraction"
        output_dir.mkdir(parents=True, exist_ok=True)

        cache_stats = text_cache.get_stats()
        metadata = {
            "extraction_timestamp": datetime.now().isoformat(),
            "phase2_dir": str(phase2_dir),
            "phase3_dir": str(phase3_dir),
            "phase1_dir": str(phase1_dir),
            "original_paper": {
                "canonical_id": original_canonical_id,
                "title": original_paper.get("title")
            },
            "statistics": {
                "total_papers": len(papers_to_download),
                "already_cached": cached_count,
                "downloaded": len(papers_to_fetch),
                "pdf_success": sum(1 for _, src, _ in download_results.values() if src in ("pdf_full", "fulltext", "constructed_pdf")),
                "abstract_fallback": sum(1 for _, src, _ in download_results.values() if src == "abstract" or src == "abstract_fallback"),
                "errors": sum(1 for _, src, _ in download_results.values() if src == "error"),
            },
            "cache_stats": cache_stats,
            "config": {
                "max_workers": max_workers,
                "force_refresh": force_refresh
            }
        }

        metadata_file = output_dir / "metadata.json"
        metadata_file.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding='utf-8'
        )
        logger.info(f"Metadata saved to: {metadata_file}")

        # Step 9: Cleanup PDFs
        logger.info("\n[Step 9] Cleaning up temporary PDFs...")
        deleted_count = cleanup_temp_pdfs(logger)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("PDF DOWNLOAD COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Total papers: {len(papers_to_download)}")
        logger.info(f"Cached: {cache_stats['total_cached_papers']} papers ({cache_stats['cache_size_mb']} MB)")
        logger.info(f"PDFs cleaned: {deleted_count}")
        logger.info(f"Output: {output_dir}")
        logger.info("=" * 80)

        return True

    except Exception as e:
        logger.error(f"PDF download failed: {e}", exc_info=True)
        return False

    finally:
        # Ensure PDFs are cleaned up even if there's an error
        try:
            cleanup_temp_pdfs(logger)
        except Exception:
            pass


def main():
    """CLI entry point for PDF download."""
    parser = argparse.ArgumentParser(
        description="Phase 3 Part 2: Download PDFs and extract text for all candidate papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m scripts.run_phase3_pdf_download \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1

  # With more parallel workers
  python -m scripts.run_phase3_pdf_download \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1 \\
      --max-workers 5

  # Force refresh cache
  python -m scripts.run_phase3_pdf_download \\
      --phase2-dir output/paper123/phase2/final \\
      --phase3-dir output/paper123/phase3 \\
      --phase1-dir output/paper123/phase1 \\
      --force-refresh
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
        help="Phase 1 directory (e.g., output/paper123/phase1)"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
        help="Maximum parallel downloads (default: 3)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh text cache (re-download all papers)"
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

    # Run download
    success = run_pdf_download(
        phase2_dir=args.phase2_dir,
        phase3_dir=args.phase3_dir,
        phase1_dir=args.phase1_dir,
        max_workers=args.max_workers,
        force_refresh=args.force_refresh,
        log_dir=args.log_dir
    )

    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
