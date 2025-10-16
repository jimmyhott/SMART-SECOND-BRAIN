"""
Centralized logging configuration for the Smart Second Brain project.

This module provides a consistent logging setup across all components
of the project, including tests, API, and agentic workflows.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = "smart_second_brain",
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        name: Logger name (default: "smart_second_brain")
        level: Logging level (default: "INFO")
        log_file: Specific log file name (optional)
        log_dir: Directory for log files (default: "logs")
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Determine log file name
    if log_file is None:
        log_file = f"{name}.log"
    
    log_file_path = log_path / log_file
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(str(log_file_path))  # File output - convert to string
        ],
        force=True  # Override any existing configuration
    )
    
    return logging.getLogger(name)


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance with the project's configuration.
    
    Args:
        name: Logger name (optional, uses module name if not provided)
    
    Returns:
        Logger instance
    """
    if name is None:
        # Use the calling module's name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'smart_second_brain')
    
    return logging.getLogger(name)


# Default project logger
project_logger = get_logger("smart_second_brain")


def setup_test_logging(test_name: str = "tests") -> logging.Logger:
    """
    Set up logging specifically for tests.
    
    Args:
        test_name: Name for the test logger
    
    Returns:
        Test logger instance
    """
    import os
    return setup_logging(
        name=test_name,
        level=os.getenv("TEST_LOG_LEVEL", "INFO"),
        log_file=f"{test_name}.log",
        log_dir="logs"
    )


def setup_api_logging() -> logging.Logger:
    """
    Set up logging specifically for the API.
    
    Returns:
        API logger instance
    """
    import os
    return setup_logging(
        name="api",
        level=os.getenv("API_LOG_LEVEL", "INFO"),
        log_file="api.log",
        log_dir="logs"
    )


def setup_agentic_logging() -> logging.Logger:
    """
    Set up logging specifically for agentic workflows.
    
    Returns:
        Agentic logger instance
    """
    import os
    return setup_logging(
        name="agentic",
        level=os.getenv("AGENTIC_LOG_LEVEL", "INFO"),
        log_file="agentic.log",
        log_dir="logs"
    )
