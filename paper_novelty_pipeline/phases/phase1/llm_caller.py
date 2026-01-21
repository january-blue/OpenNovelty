"""
Phase 1 LLM call wrapper.

Provides a thin, robust interface for text/JSON LLM calls, including fallback
parsing and raw-response persistence. This keeps LLM invocation policy and
logging in one place while leaving prompt design to extractor classes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from paper_novelty_pipeline.utils.text_cleaning import parse_json_flexible


class Phase1LLMCaller:
    """Thin wrapper for Phase1 LLM calls with optional raw-response persistence."""

    def __init__(self, llm_client, artifacts_writer, logger: Optional[logging.Logger] = None) -> None:
        self.llm_client = llm_client
        self.artifacts_writer = artifacts_writer
        self.logger = logger or logging.getLogger(__name__)

    def call_text(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        phase1_dir: Optional[Path] = None,
        call_type: str = "unknown",
    ) -> Optional[str]:
        if not self._has_client(call_type, "text"):
            return None

        raw_response, error = self._try_generate_text(messages, max_tokens, temperature)
        parse_error = None
        if error:
            parse_error = f"LLM text call failed: {error}"
            self.logger.error("Phase1: %s", parse_error)

        self._persist(
            phase1_dir,
            call_type,
            messages,
            raw_response=raw_response,
            parsed_result=None,
            parse_error=parse_error,
        )
        return raw_response

    def call_json(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        phase1_dir: Optional[Path] = None,
        call_type: str = "unknown",
    ) -> Optional[Dict[str, Any]]:
        if not self._has_client(call_type, "JSON"):
            return None

        data = self._try_generate_json(messages, max_tokens, temperature)
        if isinstance(data, dict):
            if self._is_salvaged_wrong(data, call_type):
                return self._handle_salvaged_wrong(
                    data,
                    messages,
                    max_tokens,
                    temperature,
                    phase1_dir,
                    call_type,
                )
            self._persist(
                phase1_dir,
                call_type,
                messages,
                raw_response="<parsed_directly_by_generate_json>",
                parsed_result=data,
                parse_error=None,
            )
            return data

        return self._fallback_json_parse(
            messages,
            max_tokens,
            temperature,
            phase1_dir,
            call_type,
        )

    def _has_client(self, call_type: str, mode: str) -> bool:
        if self.llm_client:
            return True
        self.logger.error("Phase1: LLM client missing for %s call (%s)", mode, call_type)
        return False

    def _try_generate_json(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> Optional[Dict[str, Any]]:
        try:
            return self.llm_client.generate_json(
                messages, max_tokens=max_tokens, temperature=temperature
            )
        except Exception as e:
            self.logger.warning(
                "Phase1: generate_json failed (%s); falling back to raw completion.",
                e,
            )
            return None

    def _try_generate_text(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
    ) -> tuple[Optional[str], Optional[Exception]]:
        try:
            return (
                self.llm_client.generate(
                    messages, max_tokens=max_tokens, temperature=temperature
                ),
                None,
            )
        except Exception as e:
            return None, e

    def _is_salvaged_wrong(self, data: Dict[str, Any], call_type: str) -> bool:
        return "is_duplicate_variant" in data and call_type in [
            "contributions",
            "core_task",
            "query_variants",
        ]

    def _handle_salvaged_wrong(
        self,
        data: Dict[str, Any],
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        phase1_dir: Optional[Path],
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        self.logger.warning(
            "Phase1: generate_json returned salvaged duplicate format for %s; retry raw.",
            call_type,
        )
        raw_response, _ = self._try_generate_text(messages, max_tokens, temperature)
        self._persist(
            phase1_dir,
            call_type,
            messages,
            raw_response=raw_response,
            parsed_result=data,
            parse_error=(
                "generate_json returned salvaged duplicate format instead of expected format"
            ),
        )
        if not raw_response:
            return data

        parsed = parse_json_flexible(raw_response, self.logger)
        if parsed is not None:
            self.logger.info(
                "Phase1: successfully parsed %s from raw response", call_type
            )
            return parsed
        return data

    def _fallback_json_parse(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        phase1_dir: Optional[Path],
        call_type: str,
    ) -> Optional[Dict[str, Any]]:
        raw_response, error = self._try_generate_text(messages, max_tokens, temperature)
        if error:
            parse_error = f"LLM call failed: {error}"
            self.logger.error("Phase1: fallback completion failed: %s", error)
            self._persist(
                phase1_dir,
                call_type,
                messages,
                raw_response=None,
                parsed_result=None,
                parse_error=parse_error,
            )
            return None

        data = parse_json_flexible(raw_response, self.logger)
        parse_error = None
        if data is None:
            parse_error = "JSON parsing failed after all fallback attempts"

        self._persist(
            phase1_dir,
            call_type,
            messages,
            raw_response=raw_response,
            parsed_result=data,
            parse_error=parse_error,
        )
        return data

    def _persist(
        self,
        phase1_dir: Optional[Path],
        call_type: str,
        messages: List[Dict[str, str]],
        *,
        raw_response: Optional[str],
        parsed_result: Optional[Dict[str, Any]],
        parse_error: Optional[str],
    ) -> None:
        if not phase1_dir:
            return
        self.artifacts_writer.save_raw_llm_response(
            phase1_dir,
            call_type,
            messages,
            raw_response=raw_response,
            parsed_result=parsed_result,
            parse_error=parse_error,
        )
