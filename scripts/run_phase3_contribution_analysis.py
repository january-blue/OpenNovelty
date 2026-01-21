#!/usr/bin/env python3
"""
Run Phase 3 (Contribution Analysis) – per-contribution full-text comparison against TopK candidates.

Usage:
  python -m scripts.run_phase3_contribution_analysis \
    --phase1-dir output/.../phase1 \
    --phase2-dir output/.../phase2 \
    --out-dir    output/... \
    --language en \
    --max-candidates 50 \
    --contribution-indices 1,2,3 \
    --max-chars-per-context 20000 \
    --resume \
    --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from paper_novelty_pipeline.phases.phase3.comparison import ComparisonWorkflow
from paper_novelty_pipeline.phases.phase3 import LLMAnalyzer, EvidenceVerifier
from paper_novelty_pipeline.services.llm_client import create_llm_client
from paper_novelty_pipeline.phases.phase3.pdf_handler import PDFHandler


def _parse_indices(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    out: List[int] = []
    for part in s.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            try:
                ai = int(a.strip()); bi = int(b.strip())
                if ai <= bi:
                    out.extend(list(range(ai, bi + 1)))
                else:
                    out.extend(list(range(bi, ai + 1)))
            except Exception:
                continue
        else:
            try:
                out.append(int(part))
            except Exception:
                continue
    return out or None


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase3 per-contribution full-text analysis")
    ap.add_argument("--phase1-dir", type=Path, required=True)
    ap.add_argument("--phase2-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--max-candidates", type=int, default=50)
    # New preferred parameter (1-based indexing for user convenience)
    ap.add_argument("--contribution-indices", type=str, default=None, 
                    help="1-based contribution indices, e.g., '1,2,3' or '1-3' (output: contribution_1/, contribution_2/, contribution_3/)")
    # Backward-compat alias (deprecated)
    ap.add_argument("--contrib-indices", type=str, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--max-chars-per-context", type=int, default=20000)
    ap.add_argument("--concurrency", type=int, default=4, help="parallelism for fetching candidate full texts")
    ap.add_argument("--max-tokens", type=int, default=1800, help="LLM max tokens per comparison")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--engine", type=str, default="comparison", choices=["comparison"],
                    help="execution engine: 'comparison' (use Phase3 comparison pipeline)")
    ap.add_argument("--resume", action="store_true", help="skip existing per-candidate outputs")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("run_phase3_contribution_analysis")

    # prefer new param if provided, else fallback to deprecated alias
    pick = args.contribution_indices if args.contribution_indices else args.contrib_indices
    contribution_indices = _parse_indices(pick)
    
    # Convert from 1-based (user-friendly) to 0-based (internal list index)
    # User input: 1, 2, 3 -> Internal index: 0, 1, 2
    if contribution_indices:
        contribution_indices = [i - 1 for i in contribution_indices if i >= 1]
        if not contribution_indices:
            contribution_indices = None

    if args.engine != "comparison":
        log.error(f"Engine '{args.engine}' is no longer supported. Only 'comparison' engine is available.")
        log.error("Please use --engine comparison (or omit this argument as it's the default).")
        sys.exit(1)

    # Initialize TextCache (shared with textual similarity detection)
    from paper_novelty_pipeline.utils.text_cache import TextCache
    phase3_dir = args.out_dir / "phase3"
    text_cache_dir = phase3_dir / "cached_paper_texts"
    text_cache = TextCache(text_cache_dir, log)
    log.info(f"✓ TextCache initialized at: {text_cache_dir}")

    # Create dependencies (same as run_phase3_core_task_comparisons.py)
    llm_client = create_llm_client()
    evidence_verifier = EvidenceVerifier(logger=log)
    pdf_handler = PDFHandler(logger=log, output_base_dir=str(args.out_dir), text_cache=text_cache)  # ✅ Share TextCache
    llm_analyzer = LLMAnalyzer(
        llm_client=llm_client,
        logger=log,
        output_base_dir=str(args.out_dir),
        evidence_verifier=evidence_verifier,
    )

    workflow = ComparisonWorkflow(
        llm_analyzer=llm_analyzer,
        evidence_verifier=evidence_verifier,
        pdf_handler=pdf_handler,
        logger=log,
        output_base_dir=str(args.out_dir),
    )
    try:
        workflow.compare_contributions_from_final(
            phase1_dir=args.phase1_dir,
            phase2_dir=args.phase2_dir,
            out_dir=args.out_dir,
            language=args.language,
            max_candidates=args.max_candidates,
            contribution_indices=contribution_indices,
            resume=args.resume,
        )
    except Exception as e:
        log.error(f"Phase3 contribution analysis failed: {e}", exc_info=True)
        sys.exit(1)
    log.info("Phase3 per-contribution analysis finished. See phase3/contribution_analysis/")


if __name__ == "__main__":
    main()
