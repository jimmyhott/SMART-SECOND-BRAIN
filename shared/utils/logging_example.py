"""
Example usage of the centralized logging configuration.

This file demonstrates how to use the project-level logging
in different components of the Smart Second Brain project.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils.logging_config import (
    setup_logging,
    get_logger,
    setup_test_logging,
    setup_api_logging,
    setup_agentic_logging
)


def example_basic_usage():
    """Example of basic logging usage."""
    # Get a logger for this module
    logger = get_logger()
    
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # You can also use the default project logger
    from shared.utils.logging_config import project_logger
    project_logger.info("Using the default project logger")


def example_test_logging():
    """Example of test-specific logging."""
    # Set up test logging
    test_logger = setup_test_logging("my_test_suite")
    
    test_logger.info("🧪 Starting test suite")
    test_logger.info("✅ Test passed")
    test_logger.error("❌ Test failed")


def example_api_logging():
    """Example of API-specific logging."""
    # Set up API logging
    api_logger = setup_api_logging()
    
    api_logger.info("🚀 API server starting")
    api_logger.info("📡 Processing request")
    api_logger.error("💥 API error occurred")


def example_agentic_logging():
    """Example of agentic workflow logging."""
    # Set up agentic logging
    agentic_logger = setup_agentic_logging()
    
    agentic_logger.info("🤖 Agentic workflow starting")
    agentic_logger.info("🧠 Processing with LLM")
    agentic_logger.info("💾 Storing results")


def example_custom_logging():
    """Example of custom logging configuration."""
    # Set up custom logging
    custom_logger = setup_logging(
        name="custom_component",
        level="DEBUG",
        log_file="custom.log",
        log_dir="logs"
    )
    
    custom_logger.debug("Debug information")
    custom_logger.info("Custom component info")
    custom_logger.warning("Custom warning")


if __name__ == "__main__":
    print("🔧 Testing centralized logging configuration...")
    
    example_basic_usage()
    example_test_logging()
    example_api_logging()
    example_agentic_logging()
    example_custom_logging()
    
    print("✅ All logging examples completed!")
    print("📁 Check the 'logs/' directory for log files")
