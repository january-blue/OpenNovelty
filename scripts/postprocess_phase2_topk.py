#!/usr/bin/env python3
"""
Compatibility CLI wrapper for Phase2 post-processing.

The in-process implementation lives in:
  `paper_novelty_pipeline.phases.phase2.postprocess.postprocess_phase2_outputs`
"""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_novelty_pipeline.phases.phase2.postprocess import postprocess_phase2_outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-process Phase2 outputs into final TopK results.")
    ap.add_argument(
        "phase2_dir",
        type=str,
        help="Path to the phase2 directory (contains core_task_candidates.json and contributions/)",
    )
    ap.add_argument("--core-topk", type=int, default=50, help="TopK to keep for core task (default: 50)")
    ap.add_argument(
        "--contrib-topk", type=int, default=10, help="TopK to keep for each contribution (default: 10)"
    )
    ap.add_argument(
        "--final-subdir",
        type=str,
        default="final",
        help="Subdirectory name under phase2_dir to write final files (default: final)",
    )
    args = ap.parse_args()

    postprocess_phase2_outputs(
        Path(args.phase2_dir),
        core_topk=args.core_topk,
        contrib_topk=args.contrib_topk,
        final_subdir=args.final_subdir,
    )


if __name__ == "__main__":
    main()





