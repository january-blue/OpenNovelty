#!/usr/bin/env python3
"""
Run Phase 3 (Core Task Comparisons) – Compare original paper against top50 core task candidates.

Usage:
  python -m scripts.run_phase3_core_task_comparisons \
    --phase1-dir output/.../phase1 \
    --phase2-dir output/.../phase2 \
    --out-dir    output/... \
    --max-candidates 50 \
    --resume \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from paper_novelty_pipeline.phases.phase3 import ComparisonWorkflow, LLMAnalyzer, EvidenceVerifier
from paper_novelty_pipeline.services.llm_client import create_llm_client
from paper_novelty_pipeline.phases.phase3.pdf_handler import PDFHandler


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase3 core task comparisons (top50)")
    ap.add_argument("--phase1-dir", type=Path, required=True, help="Path to phase1 directory")
    ap.add_argument("--phase2-dir", type=Path, required=True, help="Path to phase2 directory (contains final/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Base output directory (phase3 will be created)")
    ap.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Maximum candidates to process (None = process all)",
    )
    ap.add_argument("--resume", action="store_true", help="Skip already processed comparisons")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("run_phase3_core_task_comparisons")

    if not args.phase1_dir.exists():
        log.error(f"Phase1 directory not found: {args.phase1_dir}")
        sys.exit(1)
    if not args.phase2_dir.exists():
        log.error(f"Phase2 directory not found: {args.phase2_dir}")
        sys.exit(1)

    # 创建依赖对象
    llm_client = create_llm_client()
    evidence_verifier = EvidenceVerifier(logger=log)
    pdf_handler = PDFHandler(logger=log, output_base_dir=str(args.out_dir))  # ✅ Add output_base_dir
    llm_analyzer = LLMAnalyzer(
        llm_client=llm_client,
        logger=log,
        output_base_dir=str(args.out_dir),
        evidence_verifier=evidence_verifier,
    )

    cmp = ComparisonWorkflow(
        llm_analyzer=llm_analyzer,
        evidence_verifier=evidence_verifier,
        pdf_handler=pdf_handler,
        logger=log,
        output_base_dir=str(args.out_dir),
    )

    try:
        result = cmp.compare_core_task(
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir,
            out_dir=args.out_dir,
            max_candidates=args.max_candidates,
            resume=args.resume,
        )
        stats = result.get("statistics", {})
        total_candidates = stats.get("total_candidates", 0)
        eligible_candidates = stats.get(
            "eligible_candidates", stats.get("sibling_candidate_count", 0)
        )
        attempted = stats.get("attempted", eligible_candidates)
        completed = stats.get("completed", stats.get("successful_comparisons", 0))
        errored_candidates = stats.get("errored", stats.get("failed_comparisons", 0))
        skipped_by_design = stats.get(
            "skipped_by_design", max(0, total_candidates - eligible_candidates)
        )
        skip_reasons = stats.get("skip_reasons", {})
        taxonomy_status = stats.get("taxonomy_status", "unknown")

        log.info("Core task comparisons completed:")
        log.info(f"  Total candidates: {total_candidates}")
        log.info(f"  Eligible siblings (taxonomy): {eligible_candidates}")
        log.info(f"  Attempted comparisons: {attempted}")
        log.info(f"  Completed comparisons: {completed}")
        log.info(f"  Errored candidates: {errored_candidates}")
        log.info(f"  Skipped by design (non-siblings or missing taxonomy): {skipped_by_design}")
        if skip_reasons:
            log.info(f"  Skip reasons: {skip_reasons}")
        log.info(f"  Taxonomy status: {taxonomy_status}")
        log.info(f"Results saved to: {args.out_dir}/phase3/core_task_comparisons/")
    except Exception as e:
        log.error(f"Phase3 core task comparisons failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
