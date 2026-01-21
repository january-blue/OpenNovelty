"""Lightweight package init for phases.

Avoid importing heavy modules at import time to prevent side effects (e.g.,
LLM config assertions) when consumers only need a subset (like Phase3 survey).
Provides lazy attributes for the common phase classes.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "ContentExtractionPhase",
    "PaperSearchPhase",
    "NoveltyComparisonPhase",
    "NoveltyReportGenerator",
    "NoveltyReportPhase",
]


def __getattr__(name: str) -> Any:  # lazy imports
    if name == "ContentExtractionPhase":
        return getattr(import_module("paper_novelty_pipeline.phases.phase1.orchestrator"), name)
    if name == "PaperSearchPhase":
        # Use archived version for backward compatibility
        return getattr(import_module("paper_novelty_pipeline.phases.phase2_searching_archive"), name)
    if name == "NoveltyComparisonPhase":
        # Clean alias to the new Phase3 comparison workflow
        from paper_novelty_pipeline.phases.phase3.comparison import ComparisonWorkflow
        return ComparisonWorkflow
    if name in ("NoveltyReportGenerator", "NoveltyReportPhase", "LightweightReportGenerator", "LightweightNoveltyReportPhase"):
        mod = import_module("phases.phase4.report_rendering")
        return getattr(mod, name if name in ("LightweightReportGenerator", "LightweightNoveltyReportPhase") else "NoveltyReportGenerator")
    raise AttributeError(name)
