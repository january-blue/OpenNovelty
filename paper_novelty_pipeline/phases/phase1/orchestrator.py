"""
Phase 1 orchestrator.

Coordinates PDF acquisition, metadata enrichment, and LLM extraction into a
single Phase 1 flow. The orchestrator keeps high-level control and delegates
domain logic to specialized helpers.
"""
import logging
from pathlib import Path
from typing import Optional, List

from paper_novelty_pipeline.models import (
    PaperInput,
    ExtractedContent,
    CoreTask,
    ContributionClaim,
)

from paper_novelty_pipeline.config import PDF_DOWNLOAD_DIR
from paper_novelty_pipeline.services.llm_client import create_llm_client
from paper_novelty_pipeline.utils.text_cleaning import sanitize_for_llm
from paper_novelty_pipeline.phases.phase1 import (
    Phase1ArtifactsWriter,
    MetadataEnricher,
    PdfTextLoader,
    ContributionsExtractor,
    PriorWorkQueryGenerator,
    QueryVariantsGenerator,
    CoreTaskExtractor,
)
from paper_novelty_pipeline.phases.phase1.llm_caller import Phase1LLMCaller

from paper_novelty_pipeline.utils.io_helpers import IOHelpers




class Phase1Orchestrator:
    """Orchestrate Phase 1: download, enrich, and extract."""

    def __init__(self):
        """Initialize the Phase1 orchestrator."""
        self.logger = logging.getLogger(__name__)
        # Create LLM client using configured model/endpoint/key (from env/config)
        # Factory reads model/endpoint/key/provider from config/env unless overridden.
        self.llm_client = create_llm_client()
        self.io_helpers = IOHelpers()
        self.io_helpers.ensure_dir_exists(PDF_DOWNLOAD_DIR)
        self.metadata_enricher = MetadataEnricher(self.logger, llm_client=self.llm_client)
        self.pdf_loader = PdfTextLoader(self.logger)
        self.artifacts_writer = Phase1ArtifactsWriter(self.logger)
        self.llm_caller = Phase1LLMCaller(self.llm_client, self.artifacts_writer, self.logger)

        # Initialize LLM extractors
        self.contributions_extractor = ContributionsExtractor(
            self.llm_client, self.artifacts_writer, self.logger
        )
        self.prior_work_query_generator = PriorWorkQueryGenerator(
            self.llm_client, self.logger
        )
        self.query_variants_generator = QueryVariantsGenerator(
            self.llm_client, self.logger
        )
        self.core_task_extractor = CoreTaskExtractor(
            self.llm_client, self.artifacts_writer, self.logger
        )

    def extract_content(
        self,
        paper: PaperInput,
        phase1_dir: Optional[Path] = None,
    ) -> Optional[ExtractedContent]:
        """
        Orchestrates the entire content extraction workflow for a single paper.

        Args:
            paper: The paper to process.

        Returns:
            An ExtractedContent object, or None if the process fails.
        """
        self.logger.info(f"Starting content extraction for paper: {paper.paper_id}")

        pdf_bundle = self.pdf_loader.load(paper)
        if not pdf_bundle:
            return None
        pdf_path = str(pdf_bundle.pdf_path)
        # Ensure per-paper Phase1 directory exists if provided
        if phase1_dir is not None:
            try:
                phase1_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Phase1: failed to create phase1_dir={phase1_dir}: {e}")

        # Step 1: Extract full text with PyPDF
        full_text_raw = pdf_bundle.raw_text
        full_text = pdf_bundle.cleaned_text
        if not full_text_raw:
            self.logger.error("PyPDF failed to extract any text from the PDF.")
            return None

        # Save full text (raw + cleaned) for downstream phases, scoped to this paper only
        if phase1_dir is not None:
            try:
                raw_text = full_text_raw.encode("utf-8", errors="replace").decode("utf-8")
                cleaned_text = full_text.encode("utf-8", errors="replace").decode("utf-8")
                self.artifacts_writer.write_fulltext(phase1_dir, raw_text, cleaned_text)
            except Exception as e:
                self.logger.warning(f"Failed to save fulltext artifacts for {paper.paper_id}: {e}")

        # Metadata enrichment (OpenReview + URL hints + text-based)
        pub_info = None
        try:
            pub_info = self.metadata_enricher.enrich(
                paper,
                full_text=full_text,
                source_url=paper.paper_id,
                original_pdf_url=getattr(paper, "original_pdf_url", ""),
                call_llm_json_fn=(
                    lambda msgs, **kw: self.llm_caller.call_json(
                        msgs, phase1_dir=phase1_dir, call_type="pub_date", **kw
                    )
                ),
                extract_title=False,
            )
            if phase1_dir is not None and pub_info is not None:
                self.artifacts_writer.write_pub_date(phase1_dir, pub_info)
        except Exception as e:
            self.logger.warning(f"Phase1: metadata enrichment failed: {e}")

        # Step 2: Extract top-level contributions from full text
        contributions = self.contributions_extractor.extract(
            full_text, paper, phase1_dir=phase1_dir,
            sanitize_fn=sanitize_for_llm,
            call_llm_json_fn=self.llm_caller.call_json,
        )
        self.prior_work_query_generator.generate(
            contributions,
            sanitize_fn=sanitize_for_llm,
            call_llm_json_fn=(
                lambda msgs, **kw: self.llm_caller.call_json(
                    msgs, phase1_dir=phase1_dir, call_type="prior_work_queries", **kw
                )
            ),
        )
        self._generate_query_variants_for_claims(contributions, phase1_dir=phase1_dir)

        if not (1 <= len(contributions) <= 3):
            self.logger.warning(
                f"Phase1: expected 1-3 contributions but extracted {len(contributions)}"
            )

        # Step 2.5: extract one-sentence core_task from full text
        core_task_text = self.core_task_extractor.extract(
            full_text, paper, phase1_dir=phase1_dir,
            sanitize_fn=sanitize_for_llm,
            call_llm_json_fn=self.llm_caller.call_json,
        )
        if not core_task_text:
            # Conservative fallback to keep non-empty core_task text
            core_task_text = "core task of the paper (unspecified)"
        self.logger.info(f"Phase1 core_task: {core_task_text}")
        core_task_variants = self.query_variants_generator.generate(
            core_task_text, max_variants=3, require_prefix=False,
            sanitize_fn=sanitize_for_llm,
            call_llm_fn=(
                lambda msgs, **kw: self.llm_caller.call_text(
                    msgs, phase1_dir=phase1_dir, call_type="core_task_query_variants", **kw
                )
            ),
        )

        # Combine data into the final model (no slots)
        final_content = ExtractedContent(
            core_task=CoreTask(text=core_task_text, query_variants=core_task_variants),
            contributions=contributions,
        )

        # Log a brief summary of extracted claims for debugging
        try:
            contribution_names = [getattr(contribution, "name", "") for contribution in contributions[:2] if getattr(contribution, "name", "")]
            preview_parts = []
            if contribution_names:
                preview_parts.append("contributions: " + ", ".join(contribution_names))
            preview = " | ".join(preview_parts)
            self.logger.info(
                f"Phase1 contributions={len(contributions)}"
                + (f" [{preview}]" if preview else "")
            )
        except Exception:
            pass


        # If title is still unreliable, fallback to extracting from full text.
        if not self.metadata_enricher.is_trustworthy_title(paper):
            try:
                extracted_title = self.metadata_enricher.extract_title_from_text(
                    full_text,
                    call_llm_fn=(
                        lambda msgs, **kw: self.llm_caller.call_text(
                            msgs, phase1_dir=phase1_dir, call_type="title_extraction", **kw
                        )
                    ),
                )
                if extracted_title:
                    paper.title = extracted_title
                    self.logger.info(
                        f"Phase1: filled title from PDF text: {extracted_title[:80]}..."
                    )
            except Exception as e:
                self.logger.warning(
                    f"Phase1: failed to extract title from full text for {paper.paper_id}: {e}"
                )

        # Extract openreview_id if available (for OpenReview papers)
        # Phase2 will generate the canonical_id (title hash)
        # Format: "openreview:ID" to maintain namespace consistency
        openreview_forum_id = getattr(paper, "openreview_forum_id", None)
        if openreview_forum_id:
            try:
                paper.openreview_id = f"openreview:{openreview_forum_id}"
                # Also set on ExtractedContent for consistency
                final_content.openreview_id = f"openreview:{openreview_forum_id}"
                self.logger.info(f"Phase1: extracted openreview_id: openreview:{openreview_forum_id}")
            except Exception as e:
                self.logger.warning(f"Phase1: failed to set openreview_id: {e}")

        self.logger.info(f"Successfully extracted content for paper {paper.paper_id}")

        # Clean up the downloaded PDF
        # self.io_helpers.delete_file(pdf_path)  <-- DISABLED to persist PDF for Phase 3
        self.logger.info(f"Persisted original PDF at: {pdf_path}")
        return final_content

    def _generate_query_variants_for_claims(
        self,
        claims: List[ContributionClaim],
        max_variants: int = 3,
        phase1_dir: Optional[Path] = None,
    ) -> None:
        """Attach paraphrased query variants to each claim for Phase2 consumption."""
        if not claims:
            return
        for claim in claims:
            seed = (
                getattr(claim, "prior_work_query", "").strip()
                or getattr(claim, "author_claim_text", "").strip()
                or getattr(claim, "description", "").strip()
                or getattr(claim, "name", "").strip()
            )
            claim.query_variants = self.query_variants_generator.generate(
                seed, max_variants=max_variants, require_prefix=True,
                sanitize_fn=sanitize_for_llm,
                call_llm_fn=(
                    lambda msgs, **kw: self.llm_caller.call_text(
                        msgs, phase1_dir=phase1_dir, call_type="claim_query_variants", **kw
                    )
                ),
            )


ContentExtractionPhase = Phase1Orchestrator

__all__ = ["Phase1Orchestrator", "ContentExtractionPhase"]

