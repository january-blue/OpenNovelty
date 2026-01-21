#!/usr/bin/env python3
"""
Run Phase 3 (Short Survey) – Step 2 (taxonomy via LLM):
- Load Phase2 final Top50
- Ask LLM to build taxonomy JSON (strict), validate
- Generate Mermaid mindmap and update phase3/json_report.json

Usage:
  python -m scripts.run_phase3_taxonomy \
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
    ap = argparse.ArgumentParser(description="Phase3 taxonomy via LLM")
    ap.add_argument("--phase2-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--log-level", type=str, default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger("run_phase3_taxonomy")

    p3 = Phase3Survey()
    taxo = p3.build_taxonomy_via_llm(args.phase2_dir, args.out_dir, language=args.language)
    if taxo is None:
        log.error("Taxonomy generation failed (likely due to LLM config). See logs.")
    else:
        log.info("Taxonomy generated and figure updated.")


if __name__ == "__main__":
    main()

