"""
Configuration module for the paper novelty evaluation pipeline.
Contains all API endpoints, authentication tokens, and settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
# Updated to point to actual project root (not package directory)
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DATA_PATH = PROJECT_ROOT / "input_data" / "papers.jsonl"
OUTPUT_DATA_PATH = PROJECT_ROOT / "output_data" / "results.jsonl"
# Allow overriding the temp PDFs directory via env to support per-run isolation
PDF_DOWNLOAD_DIR = Path(os.getenv("PDF_DOWNLOAD_DIR", str(PROJECT_ROOT / "temp_pdfs")))

# GROBID Service URL
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070/api/processFulltextDocument")

# OpenReview URL
OPENREVIEW_PDF_URL = "https://openreview.net/pdf"

# Wispaper API Configuration
WISPAPER_API_ENDPOINT = os.getenv("WISPAPER_API_ENDPOINT", "https://gateway.wispaper.ai/api/v1/search/completions")


# LLM API Configuration (global defaults)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "your_llm_api_key_here")
LLM_API_ENDPOINT = os.getenv("LLM_API_ENDPOINT", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")

# Optional: dedicated configuration for Phase 3 taxonomy generation.
# If any of these are set, taxonomy calls can use a separate key/model/endpoint
# (for example, a more stable JSON-capable endpoint) without affecting the rest
# of the pipeline.
TAXONOMY_LLM_API_KEY = os.getenv("TAXONOMY_LLM_API_KEY") or None
TAXONOMY_LLM_MODEL_NAME = os.getenv("TAXONOMY_LLM_MODEL_NAME") or None
TAXONOMY_LLM_API_ENDPOINT = os.getenv("TAXONOMY_LLM_API_ENDPOINT") or None

# Translation model (for Phase 4 Chinese translation, non-reasoning model recommended)
LLM_TRANSLATION_MODEL_NAME = os.getenv("LLM_TRANSLATION_MODEL_NAME", "gpt-4o")  # Default to gpt-4o if not set

# Maximum tokens to request from the LLM for a single completion (can be overridden via env)
# Use a conservative default to avoid provider 400s; clamp against provider cap.
# Default 8000 is a balance between rich outputs and stability for long-context calls.
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8000"))
# Provider/model hard cap for max output tokens (safe default 64k for many providers)
LLM_PROVIDER_CAP = int(os.getenv("LLM_PROVIDER_CAP", "64000"))
# Effective maximum actually used by clients
EFFECTIVE_LLM_MAX_TOKENS = min(LLM_MAX_TOKENS, LLM_PROVIDER_CAP)
# Soft guard on prompt size (character-based, coarse but cheap)
LLM_MAX_PROMPT_CHARS = int(os.getenv("LLM_MAX_PROMPT_CHARS", "250000"))

# Global soft limit on how many characters of long texts we will ever feed into
# a single LLM call as "context" (after truncating references etc.).
# All phases should respect this as the upper bound for long-context snippets.
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "200000"))

# Pipeline Configuration
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "10"))
MAX_CONTRIBUTIONS_PER_PAPER = int(os.getenv("MAX_CONTRIBUTIONS_PER_PAPER", "5"))
MAX_METHODS_PER_PAPER = int(os.getenv("MAX_METHODS_PER_PAPER", "3"))

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO") # INFO, DEBUG
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Allow disabling global file logging via LOG_FILE=none
_log_file_env = os.getenv("LOG_FILE")
if _log_file_env and _log_file_env.lower() == "none":
    LOG_FILE = None
else:
    LOG_FILE = Path(_log_file_env) if _log_file_env else PROJECT_ROOT / "pipeline.log"

LOG_MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10 MB
LOG_BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

# Retry Configuration
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "30"))  # Increased for high concurrency scenarios
# PDF download specific retry limits (avoid extremely long SSL retry loops)
PDF_DOWNLOAD_MAX_RETRIES = int(os.getenv("PDF_DOWNLOAD_MAX_RETRIES", "5"))
PDF_DOWNLOAD_BACKOFF = float(os.getenv("PDF_DOWNLOAD_BACKOFF", "1.5"))
RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))  # seconds

# Phase 2 Configuration
# - PHASE2_QUERY_CONCURRENCY: per-paper query concurrency (threads). Recommend 1 when batch-running many papers.
# - PHASE2_REQUIRE_100_SUCCESS: if true, abort this paper when any query fails (do not run postprocess).
# - PHASE2_STRICT_REQUIRE_RAW_PATH: if true, treat a query as failed when search_structured returns raw_path=None
#   (typical for network/SSL failures). This avoids silently producing incomplete outputs.
PHASE2_QUERY_CONCURRENCY = int(os.getenv("PHASE2_QUERY_CONCURRENCY", "1"))
PHASE2_REQUIRE_100_SUCCESS = os.getenv("PHASE2_REQUIRE_100_SUCCESS", "true").lower() == "true"
PHASE2_STRICT_REQUIRE_RAW_PATH = os.getenv("PHASE2_STRICT_REQUIRE_RAW_PATH", "true").lower() == "true"
# Per-query retry attempts inside `search_structured` (kept separate from MAX_RETRIES to avoid extremely long waits).
PHASE2_MAX_QUERY_ATTEMPTS = int(os.getenv("PHASE2_MAX_QUERY_ATTEMPTS", "8"))

# Timeout Configuration
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "240"))  # seconds
SSE_TIMEOUT = int(os.getenv("SSE_TIMEOUT", "240"))  # seconds for Server-Sent Events

# Phase 3 Configuration
# None (unset) means compare all candidates; set to an integer to limit to top-k.
_p3_k_env = os.getenv("PHASE3_TOP_K")
PHASE3_TOP_K = int(_p3_k_env) if (_p3_k_env and _p3_k_env.isdigit()) else None
# Phase 3 Concurrency: number of concurrent candidate comparisons
# API provider recommends 200+ for GPT models, 1000+ for non-reasoning models
PHASE3_CONCURRENCY = int(os.getenv("PHASE3_CONCURRENCY", "60"))

# Phase 3 File Retention Configuration
PHASE3_KEEP_EXTRACTED_TEXT = os.getenv("PHASE3_KEEP_EXTRACTED_TEXT", "true").lower() == "true"
PHASE3_KEEP_DOWNLOADED_PDFS = os.getenv("PHASE3_KEEP_DOWNLOADED_PDFS", "true").lower() == "true"
PHASE3_CLEANUP_AFTER_DAYS = int(os.getenv("PHASE3_CLEANUP_AFTER_DAYS", "30"))  # 30 days retention

# Phase 3 Evidence Verifier Configuration
EVIDENCE_COVERAGE_WEIGHT = float(os.getenv("EVIDENCE_COVERAGE_WEIGHT", "0.7"))
EVIDENCE_HIT_RATIO_WEIGHT = float(os.getenv("EVIDENCE_HIT_RATIO_WEIGHT", "0.3"))
EVIDENCE_COMPACT_PENALTY = float(os.getenv("EVIDENCE_COMPACT_PENALTY", "0.5"))
EVIDENCE_MIN_CONFIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_MIN_CONFIDENCE_THRESHOLD", "0.6"))
EVIDENCE_PARTIAL_MATCH_THRESHOLD = float(os.getenv("EVIDENCE_PARTIAL_MATCH_THRESHOLD", "0.5"))
EVIDENCE_CACHE_MAX_SIZE = int(os.getenv("EVIDENCE_CACHE_MAX_SIZE", "10"))
EVIDENCE_CACHE_KEEP_SIZE = int(os.getenv("EVIDENCE_CACHE_KEEP_SIZE", "5"))
EVIDENCE_MIN_ANCHOR_CHARS = int(os.getenv("EVIDENCE_MIN_ANCHOR_CHARS", "20"))
EVIDENCE_MIN_ANCHOR_COVERAGE = float(os.getenv("EVIDENCE_MIN_ANCHOR_COVERAGE", "0.6"))
EVIDENCE_MAX_GAP_TOKENS = int(os.getenv("EVIDENCE_MAX_GAP_TOKENS", "300"))
EVIDENCE_MIN_BLOCK_TOKENS = int(os.getenv("EVIDENCE_MIN_BLOCK_TOKENS", "8"))

# Phase 3 LLM / Similarity Configuration
PHASE3_MIN_CONTEXT_CHARS = int(os.getenv("PHASE3_MIN_CONTEXT_CHARS", "30000"))
PHASE3_MIN_LLM_TOKENS = int(os.getenv("PHASE3_MIN_LLM_TOKENS", "5000"))
PHASE3_MAX_LLM_TOKENS = int(os.getenv("PHASE3_MAX_LLM_TOKENS", "8000"))
PHASE3_MIN_SIMILARITY_WORDS = int(os.getenv("PHASE3_MIN_SIMILARITY_WORDS", "30"))
PHASE3_MAX_SIMILARITY_SEGMENTS = int(os.getenv("PHASE3_MAX_SIMILARITY_SEGMENTS", "3"))

# Phase 3 Taxonomy / Narrative Configuration
PHASE3_MAX_LEAF_CAPACITY = int(os.getenv("PHASE3_MAX_LEAF_CAPACITY", "7"))
PHASE3_MAX_TOKENS_TAXONOMY = int(os.getenv("PHASE3_MAX_TOKENS_TAXONOMY", "6000"))
PHASE3_MAX_TOKENS_NARRATIVE = int(os.getenv("PHASE3_MAX_TOKENS_NARRATIVE", "900"))
PHASE3_MAX_TOKENS_APPENDIX = int(os.getenv("PHASE3_MAX_TOKENS_APPENDIX", "4000"))
PHASE3_ONE_LINER_MIN_WORDS = int(os.getenv("PHASE3_ONE_LINER_MIN_WORDS", "20"))
PHASE3_ONE_LINER_MAX_WORDS = int(os.getenv("PHASE3_ONE_LINER_MAX_WORDS", "30"))
PHASE3_MAX_ABSTRACT_LENGTH = int(os.getenv("PHASE3_MAX_ABSTRACT_LENGTH", "1800"))

# Validation - Bearer token no longer required as OAuth is used
assert LLM_API_KEY != "your_llm_api_key_here", "Please set LLM_API_KEY environment variable"
