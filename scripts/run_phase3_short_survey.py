#!/usr/bin/env python3
"""
Run Phase 3 (Short Survey) – Step 1 (no LLM):
- Read Phase2 final outputs
- Read Phase1 paper info
- Build initial json_report.json with placeholders

Usage:
  python -m scripts.run_phase3_short_survey \
    --phase2-dir output/.../phase2 \
    --out-dir    output/... \
    --language en --max-contrib 3 --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper_novelty_pipeline.phases.phase3.survey import Phase3Survey


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Phase 3 Short Survey (Step 1, no LLM)")
    ap.add_argument("--phase2-dir", type=Path, required=True, help="Path to phase2 directory (contains final/")
    ap.add_argument("--out-dir", type=Path, required=True, help="Base output directory (phase3 will be created)")
    ap.add_argument("--language", type=str, default="en", help="Language for survey text (en/zh)")
    ap.add_argument("--max-contrib", type=int, default=3, help="Select up to N contributions for novelty comparisons")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("run_phase3_short_survey")

    phase3 = Phase3Survey()
    report = phase3.build_initial_report(
        phase2_dir=args.phase2_dir,
        out_dir=args.out_dir,
        language=args.language,
        max_contrib=args.max_contrib,
    )

    log.info(f"Phase3 Short Survey (Step 1) finished. Wrote phase3/core_task_survey/survey_report.json under {args.out_dir}")


if __name__ == "__main__":
    main()
