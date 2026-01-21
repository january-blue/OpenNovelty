"""
I/O helper utilities for file operations.
Provides functions for reading and writing JSONL files and other common formats.
"""

import json
import logging
import os
from typing import List, Dict, Any, Iterator, Optional
from pathlib import Path


class IOHelpers:
    """Utility class for file I/O operations."""
    
    def __init__(self):
        """Initialize IO helpers."""
        self.logger = logging.getLogger(__name__)

    def ensure_dir_exists(self, dir_path: Path):
        """
        Ensures that a directory exists. If not, it creates the directory.

        Args:
            dir_path: The path to the directory.
        """
        if not dir_path.exists():
            self.logger.info(f"Creating directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read JSONL (JSON Lines) file and return list of dictionaries.
        
        Args:
            file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries, one for each line
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
        """
        self.logger.info(f"Reading JSONL file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        data = []
        line_number = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line_number += 1
                    line = line.strip()
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Skip comment lines (starting with #)
                    if line.startswith('#'):
                        continue
                    
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_number}: {e}")
                        self.logger.warning(f"Problematic line: {line[:100]}...")
                        continue
            
            self.logger.info(f"Successfully read {len(data)} records from {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading JSONL file {file_path}: {e}")
            raise
    
    def write_jsonl(self, file_path: str, data: List[Dict[str, Any]], append: bool = False) -> bool:
        """
        Write list of dictionaries to JSONL file.
        
        Args:
            file_path: Path to the output JSONL file
            data: List of dictionaries to write
            append: If True, append to existing file; if False, overwrite
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Writing JSONL file: {file_path} (append: {append})")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Determine file mode
            mode = 'a' if append else 'w'
            
            records_written = 0
            with open(file_path, mode, encoding='utf-8') as file:
                for record in data:
                    try:
                        # Convert to JSON string
                        json_line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                        file.write(json_line + '\n')
                        records_written += 1
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Failed to serialize record: {e}")
                        continue
            
            self.logger.info(f"Successfully wrote {records_written} records to {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing JSONL file {file_path}: {e}")
            return False
    
    def delete_file(self, file_path: str) -> bool:
        """
        Deletes a file if it exists.

        Args:
            file_path: The path to the file to delete.

        Returns:
            True if the file was deleted or didn't exist, False if an error occurred.
        """
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                self.logger.info(f"Successfully deleted file: {file_path}")
                return True
            except OSError as e:
                self.logger.error(f"Error deleting file {file_path}: {e}")
                return False
        else:
            self.logger.info(f"File not found, no need to delete: {file_path}")
            return True
    
    def read_json(self, file_path: str) -> Dict[str, Any]:
        """
        Read JSON file and return dictionary.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Dictionary containing the JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If JSON parsing fails
        """
        self.logger.info(f"Reading JSON file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            
            self.logger.info(f"Successfully read JSON file: {file_path}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading JSON file {file_path}: {e}")
            raise
    
    def write_json(self, file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
        """
        Write dictionary to JSON file.
        
        Args:
            file_path: Path to the output JSON file
            data: Dictionary to write
            indent: JSON formatting indentation
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Writing JSON file: {file_path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=indent)
            
            self.logger.info(f"Successfully wrote JSON file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing JSON file {file_path}: {e}")
            return False
    
    def read_text(self, file_path: str) -> str:
        """
        Read text file and return content as string.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.logger.info(f"Reading text file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            self.logger.info(f"Successfully read text file: {file_path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading text file {file_path}: {e}")
            raise
    
    def write_text(self, file_path: str, content: str) -> bool:
        """
        Write string content to text file.
        
        Args:
            file_path: Path to the output text file
            content: String content to write
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Writing text file: {file_path}")
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(content)
            
            self.logger.info(f"Successfully wrote text file: {file_path} ({len(content)} characters)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing text file {file_path}: {e}")
            return False
    
    def stream_jsonl(self, file_path: str) -> Iterator[Dict[str, Any]]:
        """
        Stream JSONL file line by line (memory efficient for large files).
        
        Args:
            file_path: Path to the JSONL file
            
        Yields:
            Dictionary for each valid line
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        self.logger.info(f"Streaming JSONL file: {file_path}")
        
        if not os.path.exists(file_path):
            self.logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        line_number = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line in file:
                    line_number += 1
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    try:
                        record = json.loads(line)
                        yield record
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON on line {line_number}: {e}")
                        continue
            
            self.logger.info(f"Finished streaming JSONL file: {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error streaming JSONL file {file_path}: {e}")
            raise
    
    def file_exists(self, file_path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(file_path)
    
    def ensure_directory(self, directory_path: str) -> bool:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            True if directory exists or was created successfully
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Error creating directory {directory_path}: {e}")
            return False
    
    def get_file_size(self, file_path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        try:
            return os.path.getsize(file_path)
        except (OSError, FileNotFoundError):
            return 0
    
    def list_files(self, directory_path: str, pattern: str = "*") -> List[str]:
        """
        List files in directory matching pattern.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match (e.g., "*.json", "*.txt")
            
        Returns:
            List of matching file paths
        """
        try:
            from pathlib import Path
            
            path = Path(directory_path)
            if pattern == "*":
                files = [str(f) for f in path.iterdir() if f.is_file()]
            else:
                files = [str(f) for f in path.glob(pattern)]
            
            self.logger.info(f"Found {len(files)} files in {directory_path} matching '{pattern}'")
            return files
            
        except Exception as e:
            self.logger.error(f"Error listing files in {directory_path}: {e}")
            return []
    
    def backup_file(self, file_path: str, backup_suffix: str = ".backup") -> Optional[str]:
        """
        Create a backup copy of a file.
        
        Args:
            file_path: Path to the original file
            backup_suffix: Suffix to add to backup filename
            
        Returns:
            Path to backup file if successful, None otherwise
        """
        if not self.file_exists(file_path):
            self.logger.warning(f"Cannot backup non-existent file: {file_path}")
            return None
        
        try:
            backup_path = file_path + backup_suffix
            
            # Read original file
            if file_path.endswith('.json'):
                data = self.read_json(file_path)
                self.write_json(backup_path, data)
            elif file_path.endswith('.jsonl'):
                data = self.read_jsonl(file_path)
                self.write_jsonl(backup_path, data)
            else:
                content = self.read_text(file_path)
                self.write_text(backup_path, content)
            
            self.logger.info(f"Created backup: {backup_path}")
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Error creating backup for {file_path}: {e}")
            return None


# Create a global instance for convenience
io_helpers = IOHelpers()

# Convenience functions that use the global instance
def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Read JSONL file using global IO helpers instance."""
    return io_helpers.read_jsonl(file_path)


def write_jsonl(file_path: str, data: List[Dict[str, Any]], append: bool = False) -> bool:
    """Write JSONL file using global IO helpers instance."""
    return io_helpers.write_jsonl(file_path, data, append)


def read_json(file_path: str) -> Dict[str, Any]:
    """Read JSON file using global IO helpers instance."""
    return io_helpers.read_json(file_path)


def write_json(file_path: str, data: Dict[str, Any], indent: int = 2) -> bool:
    """Write JSON file using global IO helpers instance."""
    return io_helpers.write_json(file_path, data, indent)


def read_text(file_path: str) -> str:
    """Read text file using global IO helpers instance."""
    return io_helpers.read_text(file_path)


def write_text(file_path: str, content: str) -> bool:
    """Write text file using global IO helpers instance."""
    return io_helpers.write_text(file_path, content)


def stream_jsonl(file_path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL file using global IO helpers instance."""
    yield from io_helpers.stream_jsonl(file_path)


def file_exists(file_path: str) -> bool:
    """Check if file exists using global IO helpers instance."""
    return io_helpers.file_exists(file_path)


def ensure_directory(directory_path: str) -> bool:
    """Ensure directory exists using global IO helpers instance."""
    return io_helpers.ensure_directory(directory_path)


def get_file_size(file_path: str) -> int:
    """Get file size using global IO helpers instance."""
    return io_helpers.get_file_size(file_path)


def list_files(directory_path: str, pattern: str = "*") -> List[str]:
    """List files using global IO helpers instance."""
    return io_helpers.list_files(directory_path, pattern)


def backup_file(file_path: str, backup_suffix: str = ".backup") -> Optional[str]:
    """Backup file using global IO helpers instance."""
    return io_helpers.backup_file(file_path, backup_suffix)