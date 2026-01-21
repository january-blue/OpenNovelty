"""
Logging configuration and utilities.
Provides centralized logging setup for the entire pipeline.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

from paper_novelty_pipeline.config import LOG_LEVEL, LOG_FORMAT, LOG_FILE, LOG_MAX_BYTES, LOG_BACKUP_COUNT


def setup_logger(
    name: Optional[str] = None,
    level: str = None,
    log_file: str = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = None,
    backup_count: int = None
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name (use __name__ for module-specific loggers)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        console_output: Whether to output to console
        file_output: Whether to output to file
        max_bytes: Maximum bytes per log file (for rotation)
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Use defaults from config if not provided
    if level is None:
        level = LOG_LEVEL
    if log_file is None:
        log_file = LOG_FILE
    if max_bytes is None:
        max_bytes = LOG_MAX_BYTES
    if backup_count is None:
        backup_count = LOG_BACKUP_COUNT
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if file_output and log_file:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Use rotating file handler to prevent huge log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, continue with console only
            print(f"Warning: Could not set up file logging: {e}")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance. If no name is provided, returns the root logger.
    
    Args:
        name: Logger name (use __name__ for module-specific loggers)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: Optional[str] = None) -> None:
    """
    Set log level for a specific logger or all loggers.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger name, or None for root logger
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Also update handlers
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))


def add_file_handler(log_file: str, logger_name: Optional[str] = None, level: str = None) -> bool:
    """
    Add a file handler to an existing logger.
    
    Args:
        log_file: Path to the log file
        logger_name: Logger name (use __name__ for module-specific loggers)
        level: Log level for this handler
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = logging.getLogger(logger_name)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        
        # Create file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding='utf-8'
        )
        
        if level:
            file_handler.setLevel(getattr(logging, level.upper()))
        else:
            file_handler.setLevel(logger.level)
            
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return True
        
    except Exception as e:
        print(f"Error adding file handler: {e}")
        return False


def add_console_handler(logger_name: Optional[str] = None, level: str = None) -> bool:
    """
    Add a console handler to an existing logger.
    
    Args:
        logger_name: Logger name (use __name__ for module-specific loggers)
        level: Log level for this handler
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger = logging.getLogger(logger_name)
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        if level:
            console_handler.setLevel(getattr(logging, level.upper()))
        else:
            console_handler.setLevel(logger.level)
            
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return True
        
    except Exception as e:
        print(f"Error adding console handler: {e}")
        return False


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        
        logger = logging.getLogger(func.__module__)
        logger.info(f"Starting execution of {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"Completed execution of {func.__name__} in {execution_time:.2f} seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed execution of {func.__name__} after {execution_time:.2f} seconds: {e}")
            raise
    
    return wrapper


def log_function_calls(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Usage:
        @log_function_calls
        def my_function(arg1, arg2):
            return arg1 + arg2
    """
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        
        # Log function call with arguments
        args_str = ', '.join([repr(arg) for arg in args])
        kwargs_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args = ', '.join([args_str, kwargs_str]) if args_str and kwargs_str else args_str or kwargs_str
        
        logger.debug(f"Calling {func.__name__}({all_args})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned: {repr(result)}")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} raised exception: {e}")
            raise
    
    return wrapper


class LogContext:
    """
    Context manager for temporary log level changes.
    
    Usage:
        with LogContext(level='DEBUG'):
            # Code here runs with DEBUG log level
            pass
    """
    
    def __init__(self, level: str, logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name
        self.original_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        set_log_level(self.level, self.logger_name)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.original_level is not None:
            logger = logging.getLogger(self.logger_name)
            logger.setLevel(self.original_level)


def setup_pipeline_logging(log_dir: str = "logs", log_prefix: str = "paper_novelty_pipeline") -> logging.Logger:
    """
    Set up comprehensive logging for the paper novelty pipeline.
    
    Args:
        log_dir: Directory for log files
        log_prefix: Prefix for log file names
        
    Returns:
        Configured root logger
    """
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{log_prefix}_{timestamp}.log")
    
    # Set up main logger
    main_logger = setup_logger(
        name=None,  # Root logger
        level=LOG_LEVEL,
        log_file=log_file,
        console_output=True,
        file_output=True
    )
    
    # Set up separate loggers for different components
    components = [
        'wispaper_client',
        'llm_client', 
        'website_client',
        'pipeline',
        'extraction',
        'search',
        'comparison',
        'upload'
    ]
    
    for component in components:
        component_log_file = os.path.join(log_dir, f"{log_prefix}_{component}_{timestamp}.log")
        setup_logger(
            name=component,
            level=LOG_LEVEL,
            log_file=component_log_file,
            console_output=False,  # Only file output for component logs
            file_output=True
        )
    
    main_logger.info(f"Pipeline logging initialized. Main log file: {log_file}")
    return main_logger


# Initialize default logger when module is imported
default_logger = setup_logger()

# Convenience functions
def info(message: str, *args, **kwargs):
    """Log info message using default logger."""
    default_logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs):
    """Log warning message using default logger."""
    default_logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs):
    """Log error message using default logger."""
    default_logger.error(message, *args, **kwargs)


def debug(message: str, *args, **kwargs):
    """Log debug message using default logger."""
    default_logger.debug(message, *args, **kwargs)


def critical(message: str, *args, **kwargs):
    """Log critical message using default logger."""
    default_logger.critical(message, *args, **kwargs)
