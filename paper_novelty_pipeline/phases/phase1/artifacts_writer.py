"""
Phase 1 artifacts writer.

Persists Phase 1 outputs (fulltext, pub date, raw LLM responses) into a stable
file layout for downstream phases and debugging. The writer centralizes file
I/O and error handling so orchestration code stays clean.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Phase1ArtifactsWriter:
    """Persist Phase1 artifacts (fulltext, LLM raw responses, and metadata)."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    def write_fulltext(self, phase1_dir: Path, raw_text: str, cleaned_text: str) -> None:
        try:
            phase1_dir.mkdir(parents=True, exist_ok=True)
            (phase1_dir / "fulltext_raw.txt").write_text(raw_text, encoding="utf-8")
            (phase1_dir / "fulltext_cleaned.txt").write_text(cleaned_text, encoding="utf-8")
        except Exception as e:
            self.logger.debug("Phase1: failed to persist fulltext debug files in %s: %s", phase1_dir, e)

    def write_body_for_claims(self, phase1_dir: Path, body: str) -> None:
        try:
            (phase1_dir / "body_for_claims.txt").write_text(body, encoding="utf-8")
        except Exception as e:
            self.logger.debug("Phase1: failed to persist body_for_claims.txt in %s: %s", phase1_dir, e)

    def write_body_for_core_task(self, phase1_dir: Path, body: str) -> None:
        try:
            (phase1_dir / "body_for_core_task.txt").write_text(body, encoding="utf-8")
        except Exception as e:
            self.logger.debug("Phase1: failed to persist body_for_core_task.txt in %s: %s", phase1_dir, e)

    def write_pub_date(self, phase1_dir: Path, pub_info: Dict[str, Any]) -> None:
        try:
            with open(phase1_dir / "pub_date.json", "w", encoding="utf-8") as f:
                json.dump(pub_info, f, ensure_ascii=False, indent=2)
            self.logger.info("Phase1: wrote pub_date.json to %s", phase1_dir)
        except Exception as e:
            self.logger.warning("Phase1: failed to write pub_date.json to %s: %s", phase1_dir, e)

    def save_raw_llm_response(
        self,
        phase1_dir: Path,
        call_type: str,
        messages: List[Dict[str, str]],
        raw_response: Optional[str],
        parsed_result: Optional[Dict[str, Any]],
        parse_error: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Save raw LLM response to disk for debugging and auditing.
        
        For text-mode calls (no JSON parsing expected), stores only the raw response.
        For JSON-mode calls, includes parsing results and error information.
        
        Args:
            phase1_dir: Phase1 output directory
            call_type: Type of LLM call (e.g., "contributions", "claim_query_variants")
            messages: Messages sent to LLM
            raw_response: Raw text response from LLM
            parsed_result: Parsed JSON result (None for text-mode calls)
            parse_error: Error message if parsing failed
            
        Returns:
            Path to saved file, or None if save failed
        """
        try:
            raw_dir = phase1_dir / "raw_llm_responses"
            raw_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            content_hash = hashlib.md5((raw_response or "").encode("utf-8")).hexdigest()[:8]
            filename = f"{call_type}_{timestamp}_{content_hash}.json"
            filepath = raw_dir / filename

            # Determine call mode: text-mode calls don't expect JSON parsing
            is_text_mode = parsed_result is None and parse_error is None
            
            # Base payload common to both modes
            payload = {
                "schema_version": 1,
                "timestamp": datetime.now().isoformat(),
                "call_type": call_type,
                "call_mode": "text" if is_text_mode else "json",
                "messages_summary": [
                    {"role": message.get("role", ""), "content_length": len(message.get("content", ""))}
                    for message in messages
                ],
                "raw_response": raw_response,
                "raw_response_length": len(raw_response) if raw_response else 0,
            }
            
            # Add JSON-specific fields only for json-mode calls
            if not is_text_mode:
                payload["parsed_result"] = parsed_result
                payload["parse_success"] = parsed_result is not None
                payload["parse_error"] = parse_error

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            self.logger.debug("Phase1: saved raw LLM response to %s", filepath)
            return filepath
        except Exception as e:
            self.logger.warning("Phase1: failed to save raw LLM response: %s", e)
            return None
