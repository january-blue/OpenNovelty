#!/usr/bin/env python3
"""
Generate Phase 3 complete report – merge all Phase3 components into a single JSON report.

Usage (all components are included by default):

  python -m scripts.run_phase3_generate_complete_report \\
    --out-dir output/... \\
    --log-level INFO
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from paper_novelty_pipeline.phases.phase3.report_generator import ReportGenerator

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Phase3 complete report")
    ap.add_argument("--out-dir", type=Path, required=True, help="Base output directory (contains phase3/)")
    ap.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("run_phase3_generate_complete_report")

    if not args.out_dir.exists():
        log.error(f"Output directory not found: {args.out_dir}")
        sys.exit(1)

    report_generator = ReportGenerator(
        logger=log,
        output_base_dir=str(args.out_dir)
    )
    try:
        complete_report = report_generator.generate_complete_report(
            out_dir=args.out_dir,
            include_survey=True,
            include_contribution=True,
            include_core_task=True,
        )
        report_path = args.out_dir / "phase3" / "phase3_complete_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(complete_report, f, ensure_ascii=False, indent=2)
        log.info(f"✅ Complete report generated: {report_path}")
        log.info("   Components included:")
        log.info("     - Core Task Survey")
        log.info("     - Contribution Analysis")
        log.info("     - Core Task Comparisons")
    except Exception as e:
        log.error(f"Failed to generate complete report: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

