"""
LLM API client for content extraction and paper comparison.
Refactored to use the official OpenAI Python SDK and support JSON mode.
"""

import ast
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore
import requests

from paper_novelty_pipeline.config import (
    LLM_API_KEY, LLM_MODEL_NAME, API_TIMEOUT, MAX_RETRIES, LLM_API_ENDPOINT,
    EFFECTIVE_LLM_MAX_TOKENS, RETRY_DELAY, LLM_PROVIDER,
)
from paper_novelty_pipeline.utils.text_cleaning import sanitize_unicode


class BaseLLMClient:
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model_name: str, base_url: str):
        """Initialize the LLM client."""
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.logger.info(f"{self.__class__.__name__} initialized with model: {model_name}")

    def generate(self, messages: List[Dict[str, str]], *, max_tokens: int = 2000, temperature: float = 0.7, use_cache: bool = False, cache_ttl: str = "1h") -> str:
        """Bare-metal text generation. Subclass must implement."""
        raise NotImplementedError

    def generate_json(self, messages: List[Dict[str, str]], *, max_tokens: int = 2000, temperature: float = 0.1, use_cache: bool = False, cache_ttl: str = "1h") -> Optional[Dict[str, Any]]:
        """Bare-metal JSON generation. Subclass must implement."""
        raise NotImplementedError

    def _effective_max_tokens(self, requested: Optional[int]) -> int:
        """Clamp requested max_tokens against a safe provider cap.

        - Uses EFFECTIVE_LLM_MAX_TOKENS from config as an upper bound.
        - Falls back to 3000 if requested is invalid.
        """
        try:
            req = int(requested) if requested is not None else 3000
        except Exception:
            req = 3000
        eff = min(req, int(EFFECTIVE_LLM_MAX_TOKENS))
        try:
            # Log once per call for traceability
            self.logger.info(f"LLM max_tokens: requested={req} -> effective={eff}")
        except Exception:
            pass
        return eff

    def _convert_to_cached_messages(self, messages: List[Dict], cache_system: bool = True, cache_ttl: str = "1h") -> List[Dict]:
        """
        Convert standard OpenAI messages to Anthropic-style cached messages.

        For system messages (when cache_system=True), converts:
            {"role": "system", "content": "..."}
        to:
            {"role": "system", "content": [{"type": "text", "text": "...", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}

        This enables prompt caching on OpenRouter with Anthropic models.

        Args:
            messages: List of message dicts
            cache_system: Whether to cache system messages
            cache_ttl: Cache TTL - "5m" (5 min, 1.25x write cost) or "1h" (1 hour, 2x write cost)
                       Cache reads are always 0.1x regardless of TTL.

        For batch processing (>5 min total), 1h TTL is more cost-effective.
        """
        converted = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system" and cache_system and isinstance(content, str):
                cache_control = {"type": "ephemeral"}
                if cache_ttl == "1h":
                    cache_control["ttl"] = "1h"

                converted.append({
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": content,
                        "cache_control": cache_control
                    }]
                })
            else:
                converted.append(msg)

        return converted

    def _preclean_text(self, text: str) -> str:
        """Remove BOM/zero-width spaces, normalize quotes, and sanitize unicode."""
        if not isinstance(text, str):
            return ""
        
        # 1. First sanitize unicode (removes surrogates that break JSON)
        t = sanitize_unicode(text)
        
        # 2. Strip BOM and zero-width chars
        t = t.replace("\ufeff", "").replace("\u200b", "").replace("\u200e", "").replace("\u200f", "")
        # 3. Normalize smart quotes to standard quotes (helps JSON parsers)
        t = t.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
        return t

    def _strip_code_fence(self, text: str) -> str:
        """Remove ```json ... ``` or ``` ... ``` fences that some models emit."""
        if not isinstance(text, str):
            return ""
        stripped = text.strip()
        if not stripped:
            return ""
        # Prefer regex extraction when possible
        try:
            m = re.match(r"^```(?:json|JSON|Json)?\s*\n?(.*?)\n?```\s*$", stripped, flags=re.DOTALL)
            if m:
                return m.group(1).strip()
        except Exception:
            pass
        # Fallback simple prefix/suffix handling
        if not stripped.startswith("```"):
            return stripped
        body = stripped[3:].lstrip()
        low = body.lower()
        if low.startswith("json"):
            body = body[4:].lstrip()
        if body.endswith("```"):
            body = body[:-3]
        return body.strip()

    def _extract_json_span(self, text: str) -> str:
        """Extract the first balanced JSON object/array substring from text using json.raw_decode."""
        if not isinstance(text, str):
            return ""
        s = text.strip()
        if not s:
            return ""
        decoder = json.JSONDecoder()
        for i, ch in enumerate(s):
            if ch in '{[':
                try:
                    _, end = decoder.raw_decode(s[i:])
                    return s[i:i+end].strip()
                except json.JSONDecodeError:
                    continue
        return s

    def _parse_json_content(self, raw_content: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        Attempt multiple strategies to coerce raw model output into JSON.
        Returns the first successfully parsed dict.
        """
        if not raw_content:
            return None
        attempts = []
        for candidate in (
            raw_content,
            self._strip_code_fence(raw_content),
            self._extract_json_span(raw_content),
        ):
            candidate = candidate.strip()
            if candidate and candidate not in attempts:
                # pre-clean each attempt
                attempts.append(self._preclean_text(candidate))

        last_error: Optional[Exception] = None
        for candidate in attempts:
            try:
                return json.loads(candidate)
            except Exception as exc:
                last_error = exc
                continue

        # One more salvage attempt: truncate to the last closing brace/bracket and parse.
        # This can rescue outputs where the model emitted trailing garbage or an
        # unterminated fragment after a valid JSON object/array.
        try:
            span = self._extract_json_span(raw_content)
            span = self._preclean_text(span or "")
            if span:
                last_brace = span.rfind("}")
                last_bracket = span.rfind("]")
                end = max(last_brace, last_bracket)
                if end != -1:
                    trimmed = span[: end + 1].strip()
                    if trimmed:
                        return json.loads(trimmed)
        except Exception as exc:
            last_error = exc

        # Final salvage attempt: some models occasionally return Python-style
        # dict literals using single quotes instead of double quotes. As a very
        # last resort, try to parse such content via ast.literal_eval and accept
        # it only if it yields a dict.
        try:
            cleaned = self._preclean_text(raw_content or "")
            py_obj = ast.literal_eval(cleaned)
            if isinstance(py_obj, dict):
                return py_obj  # type: ignore[return-value]
        except Exception:
            # Ignore and fall through to markdown salvage
            pass

        # salvage attempt: fix mismatched brace/bracket (e.g., array closed by brace)
        # This common LLM mistake happens when writing: {"key": [{...}, {...}}} 
        try:
            cleaned = self._preclean_text(raw_content or "")
            open_braces = cleaned.count('{')
            close_braces = cleaned.count('}')
            open_brackets = cleaned.count('[')
            close_brackets = cleaned.count(']')
            
            # If we have one extra } where ] should be
            if close_braces == open_braces + 1 and close_brackets == open_brackets - 1:
                last_brace = cleaned.rfind('}')
                if last_brace > 0:
                    second_last_brace = cleaned.rfind('}', 0, last_brace)
                    if second_last_brace > 0:
                        fixed = cleaned[:second_last_brace] + ']' + cleaned[second_last_brace+1:]
                        try:
                            return json.loads(fixed)
                        except Exception:
                            pass
        except Exception:
            pass

        # Markdown salvage: LLM sometimes returns markdown instead of JSON
        # Try to extract key information and build a JSON object
        try:
            result = self._salvage_markdown_response(raw_content or "")
            if result:
                self.logger.info("Salvaged JSON from markdown response")
                return result
        except Exception:
            pass

        if last_error:
            preview = raw_content[:200].replace("\n", " ")
            self.logger.error(
                f"Failed to coerce LLM response into JSON: {last_error}; head={preview!r}"
            )
        return None
    
    def _salvage_markdown_response(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Attempt to extract structured information from markdown-formatted response.
        This handles cases where LLM ignores JSON format instructions.
        
        Specifically handles core task distinction responses like:
        **Duplicate/Variant Check:** These papers are **not duplicates**...
        **Comparison:** (1) Shared taxonomy... (2) Overlapping... (3) Key differences...
        
        NOTE: This method should ONLY be used for duplicate/variant check responses.
        It will return None for responses that look like they contain other JSON structures
        (e.g., contributions, core_task) to avoid incorrect salvaging.
        """
        if not content:
            return None

        # IMPORTANT: If content looks like it contains other JSON structures,
        # don't salvage it as a duplicate check response.
        # Check for common JSON keys that indicate this is NOT a duplicate check response.
        lower_content = content.lower()
        json_structure_indicators = [
            '"contributions"',
            '"core_task"',
            '"name"',
            '"description"',
            '"author_claim_text"',
            '"query_variants"',
            '"prior_work_query"',
        ]
        for indicator in json_structure_indicators:
            if indicator in lower_content:
                # This looks like a JSON response for contributions/core_task etc.
                # Try to extract JSON from code fence instead
                json_in_fence = self._extract_json_from_code_fence(content)
                if json_in_fence:
                    return json_in_fence
                # If extraction failed, return None to let caller handle it
                return None
        
        # Only proceed with duplicate/variant salvage if content has relevant indicators
        has_duplicate_indicators = any(phrase in lower_content for phrase in [
            "duplicat", "variant", "same paper", "not the same", "comparison"
        ])
        if not has_duplicate_indicators:
            return None
        
        # Check for duplicate/variant indicators
        is_duplicate = False
        if "are duplicates" in lower_content or "is a duplicate" in lower_content:
            is_duplicate = True
        elif "not duplicates" in lower_content or "are not duplicates" in lower_content:
            is_duplicate = False
        elif "likely the same" in lower_content or "same paper" in lower_content:
            is_duplicate = True
        
        # Extract comparison text (everything after "Comparison:" or the main content)
        comparison_text = content
        
        # Try to find comparison section
        comparison_match = re.search(r'\*\*Comparison[:\*]*\*?\*?\s*(.*)', content, re.DOTALL | re.IGNORECASE)
        if comparison_match:
            comparison_text = comparison_match.group(1).strip()
        else:
            # Fallback: use content after duplicate check
            dup_match = re.search(r'(?:not duplicates|are duplicates)[^.]*\.\s*(.*)', content, re.DOTALL | re.IGNORECASE)
            if dup_match:
                comparison_text = dup_match.group(1).strip()
        
        # Clean up markdown formatting
        comparison_text = re.sub(r'\*\*([^*]+)\*\*', r'\1', comparison_text)  # Remove bold
        comparison_text = re.sub(r'\*([^*]+)\*', r'\1', comparison_text)  # Remove italic
        comparison_text = re.sub(r'\n+', ' ', comparison_text)  # Normalize newlines
        comparison_text = re.sub(r'\s+', ' ', comparison_text).strip()  # Normalize spaces
        
        # Truncate if too long (keep first ~500 chars)
        if len(comparison_text) > 600:
            comparison_text = comparison_text[:500].rsplit(' ', 1)[0] + "..."
        
        if comparison_text:
            return {
                "is_duplicate_variant": is_duplicate,
                "brief_comparison": comparison_text
            }
        
        return None

    def _extract_json_from_code_fence(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from markdown code fence.
        
        Handles formats like:
        ```json
        {"key": "value"}
        ```
        """
        if not content:
            return None

        # Try to find JSON in code fence
        # Pattern: ```json or ``` followed by JSON content followed by ```
        patterns = [
            r'```json\s*\n?(.*?)\n?```',  # ```json ... ```
            r'```\s*\n?(.*?)\n?```',       # ``` ... ```
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                json_str = match.group(1).strip()
                if json_str:
                    try:
                        # Try to parse the extracted JSON
                        parsed = json.loads(json_str)
                        if isinstance(parsed, dict):
                            self.logger.info("Extracted JSON from code fence")
                            return parsed
                    except json.JSONDecodeError:
                        # Try with preclean
                        try:
                            cleaned = self._preclean_text(json_str)
                            parsed = json.loads(cleaned)
                            if isinstance(parsed, dict):
                                self.logger.info("Extracted JSON from code fence (after preclean)")
                                return parsed
                        except Exception:
                            pass
        
        return None


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client using the official Python SDK."""

    def __init__(self, api_key: str, model_name: str, base_url: str):
        """Initialize the OpenAI client."""
        super().__init__(api_key, model_name, base_url)
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=API_TIMEOUT,
                max_retries=MAX_RETRIES,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            self.client = None

    def generate(self, messages: List[Dict[str, str]], *, max_tokens: Optional[int] = 2000, temperature: float = 0.7, use_cache: bool = False, cache_ttl: str = "1h") -> str:
        """
        Bare-metal text generation.

        Args:
            messages: OpenAI-style message list.
            max_tokens: max output tokens. If None, no limit is set.
            temperature: sampling temperature.
            use_cache: If True, enable prompt caching for system messages.
            cache_ttl: Cache TTL - "5m" or "1h" (default). Only used if use_cache=True.

        Returns:
            Generated text content or empty string on failure.
        """
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return ""
        try:
            # Convert messages for caching if requested
            effective_messages = self._convert_to_cached_messages(messages, cache_ttl=cache_ttl) if use_cache else messages
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": effective_messages,
                "temperature": temperature,
            }
            
            # Only include max_tokens if explicitly provided (not None)
            eff_max_tokens = None
            if max_tokens is not None:
                eff_max_tokens = self._effective_max_tokens(max_tokens)
                request_params["max_tokens"] = eff_max_tokens
            
            # Add timeout to prevent hanging indefinitely
            # 120 seconds should be enough for max_tokens=8000
            resp = self.client.chat.completions.create(
                **request_params,
                timeout=120.0  # 2 minutes timeout
            )
            
            # Handle unexpected response types (some APIs return string directly)
            if isinstance(resp, str):
                self.logger.warning(f"LLM returned string instead of completion object, using directly")
                return resp
            
            if not hasattr(resp, 'choices') or not resp.choices or len(resp.choices) == 0:
                self.logger.error(f"LLM response has no choices. Response type: {type(resp)}")
                return ""
            
            choice = resp.choices[0]
            content = choice.message.content
            finish_reason = getattr(choice, 'finish_reason', 'unknown')
            
            # Check for truncation
            if finish_reason == 'length':
                self.logger.warning(
                    f"LLM output was truncated (finish_reason='length'). "
                    f"Content length: {len(content)} chars. "
                    f"Requested max_tokens: {max_tokens}, effective: {eff_max_tokens if eff_max_tokens is not None else 'N/A'}. "
                    f"Consider increasing max_tokens or splitting the input."
                )
                # Still return the truncated content, but log the warning
            
            if not content:
                self.logger.error(f"LLM returned empty content. finish_reason='{finish_reason}'. Full response: {resp}")
                return ""
                
            return content
        except Exception as e:
            self.logger.error(f"generate failed: {e}")
            return ""

    def generate_json(self, messages: List[Dict[str, str]], *, max_tokens: Optional[int] = 2000, temperature: float = 0.1, use_cache: bool = False, cache_ttl: str = "1h") -> Optional[Dict[str, Any]]:
        """
        Bare-metal JSON generation.

        Args:
            messages: OpenAI-style message list.
            max_tokens: max output tokens. If None, no limit is set (API decides).
            temperature: sampling temperature.
            use_cache: If True, enable prompt caching for system messages.
            cache_ttl: Cache TTL - "5m" or "1h" (default). Only used if use_cache=True.

        Returns:
            Parsed JSON dict or None if anything goes wrong.
        """
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return None
        try:
            # Convert messages for caching if requested
            effective_messages = self._convert_to_cached_messages(messages, cache_ttl=cache_ttl) if use_cache else messages
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": effective_messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            
            # Only include max_tokens if explicitly provided (not None)
            if max_tokens is not None:
                eff_max_tokens = self._effective_max_tokens(max_tokens)
                request_params["max_tokens"] = eff_max_tokens
                self.logger.info(f"LLM max_tokens: requested={max_tokens} -> effective={eff_max_tokens}")
            else:
                self.logger.info("LLM max_tokens: not set (unlimited, API decides)")
            
            # Add timeout to prevent hanging indefinitely
            resp = self.client.chat.completions.create(
                **request_params,
                timeout=120.0  # 2 minutes timeout
            )
            
            # Handle unexpected response types (some APIs return string directly)
            if isinstance(resp, str):
                self.logger.error(f"LLM returned string instead of completion object: {resp[:200]}")
                # Try to parse it as JSON directly
                try:
                    return json.loads(resp)
                except Exception:
                    return None
            
            # Handle response structure
            if not hasattr(resp, 'choices') or not resp.choices or len(resp.choices) == 0:
                self.logger.error(f"LLM response has no choices. Response type: {type(resp)}, value: {str(resp)[:200]}")
                return None
            
            content = resp.choices[0].message.content
            finish_reason = getattr(resp.choices[0], 'finish_reason', None)
            
            if not content or not content.strip():
                # Check if this is a reasoning model that used all tokens for reasoning
                usage = getattr(resp, 'usage', None)
                reasoning_tokens = 0
                if usage:
                    usage_details = getattr(usage, 'completion_tokens_details', None)
                    if usage_details:
                        reasoning_tokens = getattr(usage_details, 'reasoning_tokens', 0)
                
                if finish_reason == 'length' and reasoning_tokens > 0:
                    self.logger.error(
                        f"LLM response content is empty because all {reasoning_tokens} tokens were used for reasoning. "
                        f"finish_reason='length'. Consider increasing max_tokens to allow for output content. "
                        f"Response structure: {resp}"
                    )
                else:
                    self.logger.error(f"LLM response content is empty. finish_reason={finish_reason}. Response structure: {resp}")
                return None
            
            # Try to parse JSON
            parsed = self._parse_json_content(content)
            if parsed is not None:
                return parsed
            self.logger.error(
                "Failed to parse LLM response as JSON even after stripping fences/spans. "
                f"Content (first 500 chars): {content[:500]}"
            )
            return None
        except Exception as e:
            self.logger.error(f"generate_json failed: {e}")
            import traceback
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None


class OpenRouterClient(BaseLLMClient):
    """Simple client for OpenRouter-style endpoints (e.g. https://openrouter.ai).

    This client issues a plain HTTP POST to {base_url}/chat/completions with
    Bearer authentication and returns either parsed JSON from the model's
    message content or the raw response dict when parsing is not possible.
    
    Supports prompt caching for Anthropic models via cache_control.
    """

    def __init__(self, api_key: str, model_name: str, base_url: str):
        super().__init__(api_key, model_name, base_url)

    def generate(self, messages: List[Dict[str, str]], *, max_tokens: int = 2000, temperature: float = 0.7, use_cache: bool = False, cache_ttl: str = "1h") -> str:
        """
        Generate text response with retry logic.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        eff_max_tokens = self._effective_max_tokens(max_tokens)
        
        # Pre-clean all user/system content to avoid surrogate issues
        cleaned_messages = []
        for msg in messages:
            cleaned_messages.append({
                "role": msg.get("role", "user"),
                "content": self._preclean_text(msg.get("content", ""))
            })
        
        # Convert messages for caching if requested
        effective_messages = self._convert_to_cached_messages(cleaned_messages, cache_ttl=cache_ttl) if use_cache else cleaned_messages
        
        payload = {
            "model": self.model_name,
            "messages": effective_messages,
            "max_tokens": eff_max_tokens,
            "temperature": temperature,
        }

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                r.raise_for_status()
                j = r.json()
                
                # Log cache statistics if available
                if use_cache:
                    usage = j.get("usage", {})
                    details = usage.get("prompt_tokens_details", {})
                    cached = details.get("cached_tokens", 0)
                    cache_write = details.get("cache_write_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    if cached > 0:
                        self.logger.info(f"📦 CACHE HIT: {cached}/{prompt_tokens} tokens from cache")
                    elif cache_write > 0:
                        self.logger.info(f"📝 CACHE WRITE: {cache_write} tokens written to cache")
                
                try:
                    content = j.get("choices", [])[0].get("message", {}).get("content")
                    return content if content else ""
                except Exception:
                    return ""
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                resp_content = ""
                try:
                    resp_content = f" | resp: {r.json()}" if 'r' in locals() and r.status_code != 200 else ""
                except Exception:
                    resp_content = f" | resp: {r.text}" if 'r' in locals() else ""
                
                self.logger.warning(f"OpenRouter generate attempt {attempt+1}/{MAX_RETRIES} failed: {e}{resp_content}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    self.logger.error(f"OpenRouter generate failed after {MAX_RETRIES} attempts.")
                    return ""
            except Exception as e:
                self.logger.error(f"OpenRouter generate unexpected error: {e}")
                return ""
        return ""

    def generate_json(self, messages: List[Dict[str, str]], *, max_tokens: int = 2000, temperature: float = 0.1, use_cache: bool = False, cache_ttl: str = "1h") -> Optional[Dict[str, Any]]:
        """
        Generate JSON response with retry logic.
        """
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        eff_max_tokens = self._effective_max_tokens(max_tokens)
        
        # Pre-clean all user/system content to avoid surrogate issues
        cleaned_messages = []
        for msg in messages:
            cleaned_messages.append({
                "role": msg.get("role", "user"),
                "content": self._preclean_text(msg.get("content", ""))
            })
        
        # Convert messages for caching if requested
        effective_messages = self._convert_to_cached_messages(cleaned_messages, cache_ttl=cache_ttl) if use_cache else cleaned_messages
        
        payload = {
            "model": self.model_name,
            "messages": effective_messages,
            "max_tokens": eff_max_tokens,
            "temperature": temperature,
        }

        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=API_TIMEOUT)
                r.raise_for_status()
                j = r.json()
                
                # Log cache statistics if available
                if use_cache:
                    usage = j.get("usage", {})
                    details = usage.get("prompt_tokens_details", {})
                    cached = details.get("cached_tokens", 0)
                    cache_write = details.get("cache_write_tokens", 0)
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    if cached > 0:
                        self.logger.info(f"📦 CACHE HIT: {cached}/{prompt_tokens} tokens from cache (saved ~{cached * 0.9:.0f} tokens cost)")
                    elif cache_write > 0:
                        self.logger.info(f"📝 CACHE WRITE: {cache_write} tokens written to cache")
                
                try:
                    content = j.get("choices", [])[0].get("message", {}).get("content")
                except Exception:
                    content = None
                if content:
                    parsed = self._parse_json_content(content)
                    if parsed is not None:
                        return parsed
                    
                # If parsing failed or no content, return None
                self.logger.error(f"OpenRouter generate_json: failed to parse content as JSON. Content preview: {str(content)[:200]}")
                return None
            except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                resp_content = ""
                try:
                    resp_content = f" | resp: {r.json()}" if 'r' in locals() and r.status_code != 200 else ""
                except Exception:
                    resp_content = f" | resp: {r.text}" if 'r' in locals() else ""

                self.logger.warning(f"OpenRouter generate_json attempt {attempt+1}/{MAX_RETRIES} failed: {e}{resp_content}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    self.logger.error(f"OpenRouter generate_json failed after {MAX_RETRIES} attempts.")
                    return None
            except Exception as e:
                self.logger.error(f"OpenRouter generate_json unexpected error: {e}")
                return None
        return None


def create_llm_client(
    model_name: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
) -> Optional[BaseLLMClient]:
    """
    Factory function for OpenAI-compatible or OpenRouter clients.

    Args:
        model_name: Override model name (defaults to config/env).
        api_endpoint: Override API endpoint (defaults to config/env).
        api_key: Override API key (defaults to config/env).
        provider: 'openai' or 'openrouter'. If omitted, falls back to LLM_PROVIDER.

    Returns:
        A concrete LLM client instance.
    """
    logger = logging.getLogger(__name__)

    # Use provided values or fall back to config
    key = api_key if api_key is not None else LLM_API_KEY
    model = model_name if model_name is not None else LLM_MODEL_NAME
    endpoint = api_endpoint if api_endpoint is not None else LLM_API_ENDPOINT

    # Normalize provider choice
    normalized_endpoint = (endpoint or '').lower()
    normalized_provider = (provider or LLM_PROVIDER or 'openai').lower()
    if 'openrouter.ai' in normalized_endpoint:
        normalized_provider = 'openrouter'

    # Ensure endpoint has proper format with /v1 suffix for OpenAI-compatible APIs
    if endpoint and not endpoint.endswith(('/v1', '/v1/')):
        if not endpoint.endswith('/'):
            endpoint = endpoint + '/'
        endpoint = endpoint + 'v1'

    # Channel 1: OpenRouter
    if normalized_provider == 'openrouter':
        logger.info(f"Using OpenRouter client with model: {model}")
        return OpenRouterClient(key, model, endpoint)

    # Channel 2: Standard OpenAI or other OpenAI-compatible APIs
    if normalized_provider in ('openai', 'azure', 'default'):
        if model and '/' in model:
            try:
                model = model.split('/')[-1]
            except Exception:
                pass
        logger.info(f"Using OpenAI client with model: {model}")
        return OpenAIClient(key, model, endpoint)

    raise ValueError(f"Unsupported LLM provider: {provider or normalized_provider}")
