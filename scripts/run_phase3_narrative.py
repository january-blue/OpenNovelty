#!/usr/bin/env python3
"""
Run Phase 3 (Short Survey) – Step 3:
- Generate narrative (overview, clusters, original_position, trends_gaps)
- Generate Top50 one-liners and merge into json_report
- Render markdown report (core_task_survey.md)

Usage:
  python -m scripts.run_phase3_narrative \
    --phase2-dir output/.../phase2 \
    --out-dir    output/... \
    --language en --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from paper_novelty_pipeline.phases.phase3.survey import Phase3Survey


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase3 narrative and one-liners")
    ap.add_argument("--phase2-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--language", type=str, default="en")
    # Note: --mode option removed as build_overview_tree_markdown is no longer available.
    # The script now always uses build_narrative_via_llm which generates a concise 2-paragraph narrative.
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("run_phase3_narrative")

    p3 = Phase3Survey()
    nar = p3.build_narrative_via_llm(
        args.phase2_dir,
        args.out_dir,
        language=args.language,
    )
    if nar is None:
        log.error("Narrative generation failed.")
        return

    liners = p3.build_appendix_via_llm(args.phase2_dir, args.out_dir, language=args.language)
    if liners is None:
        log.warning("One-liners generation failed or partial; check logs.")
    else:
        log.info("One-liners merged.")


if __name__ == "__main__":
    main()
