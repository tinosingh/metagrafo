"""Logging utilities for the application."""
import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs logs in JSON format."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_record = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
        
        # Add any extra attributes
        if hasattr(record, 'data') and isinstance(record.data, dict):
            log_record.update(record.data)
            
        return json.dumps(log_record, ensure_ascii=False)

def setup_logging(
    log_level: str = 'INFO',
    log_file: Optional[str] = None,
    console: bool = True,
    json_format: bool = False
) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to log to console
        json_format: Whether to use JSON format for logs
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if requested
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        # Create directory if it doesn't exist
        log_path = Path(log_file).parent
        log_path.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get a logger with the given name and optional extra data.
    
    Args:
        name: Logger name
        **kwargs: Additional data to include in log records
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add data to log records if provided
    if kwargs:
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **factory_kwargs):
            record = old_factory(*args, **factory_kwargs)
            record.data = kwargs  # type: ignore
            return record
            
        logging.setLogRecordFactory(record_factory)
    
    return logger

class LogExecutionTime:
    """Context manager for logging execution time of code blocks."""
    
    def __init__(self, name: str, logger: logging.Logger):
        """Initialize with a name for the code block and a logger."""
        self.name = name
        self.logger = logger
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timing when entering the context."""
        self.start_time = datetime.now().timestamp()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log execution time when exiting the context."""
        if self.start_time is not None:
            duration = datetime.now().timestamp() - self.start_time
            self.logger.info(
                "Execution time for %s: %.3f seconds", 
                self.name, duration,
                extra={"execution_time_seconds": duration}
            )

def log_execution_time(name: str, logger: logging.Logger):
    """
    Decorator to log the execution time of a function.
    
    Args:
        name: Name for the operation being timed
        logger: Logger instance to use for logging
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with LogExecutionTime(name, logger):
                return func(*args, **kwargs)
        return wrapper
    return decorator
