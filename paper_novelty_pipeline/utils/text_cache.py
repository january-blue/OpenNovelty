"""
Text Cache Module for Phase 3

Provides caching functionality for extracted paper texts to avoid redundant
PDF downloads and text extractions across different Phase 3 analysis modules.

Key Features:
- Cache extracted full texts by canonical_id
- Avoid redundant PDF downloads and extractions
- Cross-module reuse (Core Task, Contribution, Similarity Detection)
- Persistent file-based storage
"""

import json
import hashlib
import logging
import threading
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from paper_novelty_pipeline.utils.text_cleaning import sanitize_unicode


class TextCache:
    """
    Paper full text caching manager.
    
    Caches extracted paper texts to avoid redundant downloads and extractions.
    Uses canonical_id as the cache key to ensure uniqueness.
    
    Cache Structure:
        cache_dir/
        └─ <hash_of_canonical_id>.json
           {
               "canonical_id": "doi:10.48550/arxiv.2403.01460",
               "full_text": "...",
               "metadata": {...},
               "cached_at": "2024-12-24T10:30:00"
           }
    """
    
    def __init__(self, cache_dir: Path, logger: Optional[logging.Logger] = None):
        """
        Initialize text cache.
        
        Args:
            cache_dir: Directory for storing cached texts (e.g., phase3/text_cache)
            logger: Logger instance (if None, creates default logger)
        """
        self.cache_dir = Path(cache_dir)
        self.logger = logger or logging.getLogger(__name__)
        self._write_lock = threading.Lock()  # Thread-safe write operations
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"TextCache initialized at: {self.cache_dir}")
    
    def get_cached_text(self, canonical_id: str) -> Optional[str]:
        """
        Retrieve cached text for a paper.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            Cached full text, or None if not found
        """
        if not canonical_id:
            return None
        
        cache_file = self._get_cache_path(canonical_id)
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            full_text = data.get("full_text")
            
            if full_text:
                self.logger.debug(f"Cache hit for {canonical_id}")
                return full_text
            else:
                self.logger.warning(f"Cache file exists but contains no text: {canonical_id}")
                return None
        
        except Exception as e:
            self.logger.error(f"Failed to read cache for {canonical_id}: {e}")
            return None
    
    def cache_text(
        self,
        canonical_id: str,
        full_text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Cache extracted text with metadata.
        
        Args:
            canonical_id: Canonical ID of the paper
            full_text: Extracted full text
            metadata: Additional metadata (title, source_url, paper_id, etc.)
            
        Returns:
            True if caching succeeded, False otherwise
        """
        if not canonical_id or not full_text:
            self.logger.warning("Cannot cache: missing canonical_id or full_text")
            return False
        
        cache_file = self._get_cache_path(canonical_id)
        
        # Sanitize text to remove surrogates before JSON encoding
        sanitized_text = sanitize_unicode(full_text)
        
        try:
            cache_data = {
                "canonical_id": canonical_id,
                "full_text": sanitized_text,
                "metadata": metadata or {},
                "cached_at": datetime.now().isoformat(),
                "text_length": len(sanitized_text)
            }
            
            # Thread-safe write with lock
            with self._write_lock:
                cache_file.write_text(
                    json.dumps(cache_data, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
            
            self.logger.debug(f"Cached text for {canonical_id} ({len(full_text)} chars)")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to cache text for {canonical_id}: {e}")
            return False
    
    def has_cached(self, canonical_id: str) -> bool:
        """
        Check if a paper's text is cached.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            True if cached, False otherwise
        """
        if not canonical_id:
            return False
        
        cache_file = self._get_cache_path(canonical_id)
        return cache_file.exists()
    
    def get_cache_metadata(self, canonical_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a cached paper without loading the full text.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            Metadata dict, or None if not found
        """
        if not canonical_id:
            return None
        
        cache_file = self._get_cache_path(canonical_id)
        
        if not cache_file.exists():
            return None
        
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            # Return everything except full_text
            return {
                "canonical_id": data.get("canonical_id"),
                "metadata": data.get("metadata", {}),
                "cached_at": data.get("cached_at"),
                "text_length": data.get("text_length")
            }
        except Exception as e:
            self.logger.error(f"Failed to read metadata for {canonical_id}: {e}")
            return None
    
    def invalidate(self, canonical_id: str) -> bool:
        """
        Remove a paper's cached text.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            True if removed, False if not found or error
        """
        if not canonical_id:
            return False
        
        cache_file = self._get_cache_path(canonical_id)
        
        if not cache_file.exists():
            return False
        
        try:
            cache_file.unlink()
            self.logger.debug(f"Invalidated cache for {canonical_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache for {canonical_id}: {e}")
            return False
    
    def clear_all(self) -> int:
        """
        Clear all cached texts.
        
        Returns:
            Number of cache files removed
        """
        count = 0
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to remove {cache_file}: {e}")
            
            self.logger.info(f"Cleared {count} cache files")
            return count
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Statistics dict with:
            - total_cached_papers: Number of cached papers
            - cache_size_mb: Total cache size in MB
            - cache_directory: Path to cache directory
            - oldest_entry: Timestamp of oldest cached entry
            - newest_entry: Timestamp of newest cached entry
        """
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            total_papers = len(cache_files)
            
            if total_papers == 0:
                return {
                    "total_cached_papers": 0,
                    "cache_size_mb": 0.0,
                    "cache_directory": str(self.cache_dir),
                    "oldest_entry": None,
                    "newest_entry": None
                }
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Find oldest and newest entries
            timestamps = []
            for cache_file in cache_files:
                try:
                    data = json.loads(cache_file.read_text(encoding='utf-8'))
                    cached_at = data.get("cached_at")
                    if cached_at:
                        timestamps.append(cached_at)
                except Exception:
                    pass
            
            oldest = min(timestamps) if timestamps else None
            newest = max(timestamps) if timestamps else None
            
            return {
                "total_cached_papers": total_papers,
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_directory": str(self.cache_dir),
                "oldest_entry": oldest,
                "newest_entry": newest
            }
        
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {
                "total_cached_papers": 0,
                "cache_size_mb": 0.0,
                "cache_directory": str(self.cache_dir),
                "error": str(e)
            }
    
    def _get_cache_path(self, canonical_id: str) -> Path:
        """
        Generate cache file path for a canonical_id.
        
        Uses sanitized canonical_id as filename for readability.
        Falls back to checking MD5 hash format for backward compatibility.
        
        Args:
            canonical_id: Canonical ID of the paper
            
        Returns:
            Path to cache file
        """
        import re
        
        # New format: sanitize canonical_id directly
        # Examples:
        #   arxiv:2403.01460 → arxiv_2403.01460.json
        #   doi:10.48550/arxiv.2403.01460 → doi_10.48550_arxiv.2403.01460.json
        #   openreview:KAGR7Mqu4h → openreview_KAGR7Mqu4h.json
        safe_name = re.sub(r'[^\w\-.]', '_', canonical_id)
        safe_name = safe_name[:150] if len(safe_name) > 150 else safe_name
        new_format_path = self.cache_dir / f"{safe_name}.json"
        
        # Check if new format exists
        if new_format_path.exists():
            return new_format_path
        
        # Fallback: check old MD5 format for backward compatibility
        old_format_name = hashlib.md5(canonical_id.encode('utf-8')).hexdigest()
        old_format_path = self.cache_dir / f"{old_format_name}.json"
        
        if old_format_path.exists():
            # Found old format file, migrate it to new format
            try:
                self.logger.info(f"Migrating cache file to new format: {canonical_id}")
                old_format_path.rename(new_format_path)
                return new_format_path
            except Exception as e:
                self.logger.warning(f"Failed to migrate cache file for {canonical_id}: {e}")
                return old_format_path
        
        # Neither exists, return new format path for creation
        return new_format_path
    
    def get_canonical_id_from_cache_file(self, cache_file: Path) -> Optional[str]:
        """
        Extract canonical_id from a cache file.
        
        Useful for reverse lookup and debugging.
        
        Args:
            cache_file: Path to cache file
            
        Returns:
            Canonical ID, or None if file is invalid
        """
        try:
            data = json.loads(cache_file.read_text(encoding='utf-8'))
            return data.get("canonical_id")
        except Exception as e:
            self.logger.error(f"Failed to read canonical_id from {cache_file}: {e}")
            return None
    
    def list_cached_papers(self) -> list:
        """
        List all cached papers with their metadata.
        
        Returns:
            List of dicts with canonical_id, metadata, and cache info
        """
        cached_papers = []
        
        try:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    data = json.loads(cache_file.read_text(encoding='utf-8'))
                    cached_papers.append({
                        "canonical_id": data.get("canonical_id"),
                        "title": data.get("metadata", {}).get("title", ""),
                        "paper_id": data.get("metadata", {}).get("paper_id", ""),
                        "cached_at": data.get("cached_at"),
                        "text_length": data.get("text_length", 0)
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to read cache file {cache_file}: {e}")
            
            # Sort by cached_at (newest first)
            cached_papers.sort(
                key=lambda x: x.get("cached_at", ""),
                reverse=True
            )
            
            return cached_papers
        
        except Exception as e:
            self.logger.error(f"Failed to list cached papers: {e}")
            return []

