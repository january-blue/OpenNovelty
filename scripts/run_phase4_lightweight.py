#!/usr/bin/env python3
"""
Run Phase 4 Lightweight Report Generation.

Generates a lightweight novelty assessment report from phase3_complete_report.json.
Uses template-based generation with a single LLM call for overall assessment.

Usage:
    python -m scripts.run_phase4_lightweight \
        --phase3-report output/.../phase3/phase3_complete_report.json \
        --output-dir output/.../phase4 \
        [--no-pdf]
"""

import argparse
import logging
import sys
from pathlib import Path

# Import Phase 4 lightweight generator
from paper_novelty_pipeline.phases.phase4 import LightweightReportGenerator
from paper_novelty_pipeline.utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(
        description="Generate lightweight novelty assessment report"
    )
    parser.add_argument(
        "--phase3-report",
        type=str, 
        required=True,
        help="Path to phase3_complete_report.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,  
        required=True,
        help="Output directory for report (Markdown and optionally PDF)"
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation (only generate Markdown)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    phase3_report_path = Path(args.phase3_report)
    output_dir_path = Path(args.output_dir)
    
    # Setup logging
    setup_logger(level=args.log_level)
    log = logging.getLogger("run_phase4_lightweight")
    
    # Validate input file
    if not phase3_report_path.exists():
        log.error(f"Phase3 report not found: {phase3_report_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = LightweightReportGenerator(generate_pdf=not args.no_pdf)
    
    # Generate report
    try:
        log.info(f"Generating lightweight report from: {phase3_report_path}")
        result = generator.generate_report(phase3_report_path, output_dir_path)
        
        if "markdown" in result:
            log.info(f"✅ Markdown report generated: {result['markdown']}")
        else:
            log.error("Failed to generate report")
            sys.exit(1)
        
        if "pdf" in result:
            log.info(f"✅ PDF report generated: {result['pdf']}")
        elif not args.no_pdf:
            log.warning("PDF generation was requested but failed. Markdown report is available.")
        
        log.info("Report generation completed successfully!")
        
    except Exception as e:
        log.error(f"Failed to generate report: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
