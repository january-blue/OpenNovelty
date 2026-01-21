"""Phase 2: Paper searching and postprocessing.

Phase 2 responsibilities:
1. Call Wispaper API to search for related papers (searching.py)
2. Postprocessing: deduplication, ranking, generate citation_index (postprocess.py)
"""

from paper_novelty_pipeline.phases.phase2.searching import PaperSearcher
from paper_novelty_pipeline.phases.phase2.postprocess import postprocess_phase2_outputs

__all__ = [
    "PaperSearcher",
    "postprocess_phase2_outputs",
]
