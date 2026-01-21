"""
Phase 3: Novelty Comparison and Survey

This package contains modules for:
- comparison: Contribution-level novelty comparison workflow
- survey: Core task survey and taxonomy generation
- llm_analyzer: LLM-based analysis for comparisons
- evidence_verifier: Evidence verification and similarity detection
- report_generator: Report generation and statistics
- utils: Utility functions

Uses lazy imports to avoid import-time side effects (e.g., LLM config assertions).
"""

from importlib import import_module
from typing import Any

__all__ = [
    # Main workflow classes
    "Phase3Survey",
    "ComparisonWorkflow",
    
    # Component classes
    "LLMAnalyzer",
    "EvidenceVerifier",
    "ReportGenerator",
    "CitationManager",
    
    # Data classes
    "AnchorMatch",
    
    # Utility functions (commonly used)
    "sanitize_filename",
    "sanitize_id",
    "read_json_list",
    "get_comparison_filename",
    "extract_authors",
    "make_retrieved_paper",
]


def __getattr__(name: str) -> Any:
    """
    Lazy import to avoid import-time side effects.
    
    Only imports modules when they are actually accessed.
    """
    # Main workflow classes
    if name == "Phase3Survey":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.survey"), name)
    
    if name == "ComparisonWorkflow":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.comparison"), name)
    
    # Component classes
    if name == "LLMAnalyzer":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.llm_analyzer"), name)
    
    if name == "EvidenceVerifier":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.evidence_verifier"), name)
    
    if name == "ReportGenerator":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.report_generator"), name)
    
    if name == "CitationManager":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.citation_manager"), name)
    
    # Data classes
    if name == "AnchorMatch":
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.evidence_verifier"), name)
    
    # Utility functions
    if name in (
        "sanitize_filename",
        "sanitize_id",
        "parse_json_flexible",
        "read_json_list",
        "get_short_paper_id",
        "normalize_title",
        "extract_authors",
        "make_retrieved_paper",
        "get_comparison_filename",
    ):
        return getattr(import_module("paper_novelty_pipeline.phases.phase3.utils"), name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")