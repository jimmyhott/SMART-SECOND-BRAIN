"""
Example demonstrating the benefits of using pathlib for path operations.

This file shows how pathlib provides a more modern and intuitive way
to handle file and directory paths compared to os.path.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from shared.utils.logging_config import setup_logging

# Set up logging
logger = setup_logging("pathlib_example")


def demonstrate_pathlib_benefits():
    """Demonstrate the benefits of using pathlib."""
    
    # 1. Creating paths - more intuitive than os.path.join
    logger.info("🔧 Pathlib Benefits Demonstration")
    
    # Old way with os.path
    # import os
    # project_root = os.path.join(os.path.dirname(__file__), '..', '..')
    # logs_dir = os.path.join(project_root, 'logs')
    
    # New way with pathlib
    project_root = Path(__file__).parent.parent.parent
    logs_dir = project_root / 'logs'
    
    logger.info(f"📁 Project root: {project_root}")
    logger.info(f"📁 Logs directory: {logs_dir}")
    
    # 2. Path operations are more intuitive
    config_file = project_root / 'pyproject.toml'
    env_file = project_root / '.env'
    
    logger.info(f"📄 Config file exists: {config_file.exists()}")
    logger.info(f"📄 Env file exists: {env_file.exists()}")
    
    # 3. Directory operations
    logs_dir.mkdir(exist_ok=True)  # Create if doesn't exist
    logger.info(f"📁 Logs directory created/exists: {logs_dir.exists()}")
    
    # 4. File operations
    test_log = logs_dir / 'pathlib_test.log'
    test_log.write_text("This is a test log entry\n")
    logger.info(f"📄 Test log created: {test_log.exists()}")
    
    # 5. Path components
    logger.info(f"📁 File name: {test_log.name}")
    logger.info(f"📁 File stem: {test_log.stem}")
    logger.info(f"📁 File suffix: {test_log.suffix}")
    logger.info(f"📁 Parent directory: {test_log.parent}")
    
    # 6. Globbing and pattern matching
    log_files = list(logs_dir.glob('*.log'))
    logger.info(f"📁 Found {len(log_files)} log files:")
    for log_file in log_files:
        logger.info(f"   - {log_file.name}")
    
    # 7. Path resolution
    resolved_path = test_log.resolve()
    logger.info(f"📁 Resolved path: {resolved_path}")
    
    # 8. Relative paths
    relative_to_project = test_log.relative_to(project_root)
    logger.info(f"📁 Relative to project: {relative_to_project}")
    
    # 9. Path joining with different types
    subdir = logs_dir / 'subdir' / 'nested'
    subdir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Created nested directory: {subdir}")
    
    # 10. File operations with context managers
    test_file = subdir / 'test.txt'
    test_file.write_text("Hello from pathlib!")
    
    content = test_file.read_text()
    logger.info(f"📄 File content: {content}")
    
    # 11. Iterating over directory contents
    logger.info("📁 Directory contents:")
    for item in logs_dir.iterdir():
        if item.is_file():
            logger.info(f"   📄 File: {item.name}")
        elif item.is_dir():
            logger.info(f"   📁 Directory: {item.name}")
    
    # 12. Path comparison
    path1 = Path('logs/test.log')
    path2 = Path('logs') / 'test.log'
    logger.info(f"📁 Paths are equal: {path1 == path2}")
    
    # Cleanup
    test_log.unlink()  # Delete file
    test_file.unlink()
    subdir.rmdir()  # Remove empty directory
    logger.info("🧹 Cleanup completed")


def demonstrate_cross_platform_paths():
    """Demonstrate cross-platform path handling."""
    
    logger.info("🌍 Cross-platform path handling:")
    
    # Pathlib automatically handles different path separators
    windows_style = Path('logs\\test.log')
    unix_style = Path('logs/test.log')
    
    logger.info(f"📁 Windows style: {windows_style}")
    logger.info(f"📁 Unix style: {unix_style}")
    logger.info(f"📁 Normalized: {windows_style.resolve()}")
    
    # Path parts
    complex_path = Path('/home/user/project/logs/test.log')
    logger.info(f"📁 Path parts: {list(complex_path.parts)}")
    logger.info(f"📁 Drive: {complex_path.drive}")
    logger.info(f"📁 Root: {complex_path.root}")


if __name__ == "__main__":
    demonstrate_pathlib_benefits()
    demonstrate_cross_platform_paths()
    
    logger.info("✅ Pathlib demonstration completed!")
    logger.info("📚 Key benefits:")
    logger.info("   - More intuitive path operations")
    logger.info("   - Cross-platform compatibility")
    logger.info("   - Object-oriented interface")
    logger.info("   - Built-in file operations")
    logger.info("   - Pattern matching and globbing")
