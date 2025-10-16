#!/usr/bin/env python3
"""
Test runner for MasterGraphBuilder workflow tests.

This script provides an easy way to run the graph builder tests
and provides detailed output about the test results.
"""

import sys
import subprocess
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import centralized logging
from shared.utils.logging_config import setup_test_logging

# Set up test runner logging
logger = setup_test_logging("test_runner")

def run_tests():
    """Run the MasterGraphBuilder tests."""
    logger.info("🧪 Running MasterGraphBuilder Tests")
    logger.info("=" * 50)
    
    # Get the test file path
    test_file = Path(__file__).parent / "test_master_graph_builder.py"
    
    if not test_file.exists():
        logger.error(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)
        
        # Print output
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Errors/Warnings:")
            logger.warning(result.stderr)
        
        # Return success/failure
        success = result.returncode == 0
        
        if success:
            logger.info("\n✅ All tests passed!")
        else:
            logger.error(f"\n❌ Tests failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False

def run_specific_test(test_name=None):
    """Run a specific test or test class."""
    logger.info(f"🧪 Running specific test: {test_name or 'all'}")
    logger.info("=" * 50)
    
    test_file = Path(__file__).parent / "test_master_graph_builder.py"
    
    try:
        cmd = [sys.executable, "-m", "pytest", str(test_file), "-v"]
        
        if test_name:
            cmd.extend(["-k", test_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Errors/Warnings:")
            logger.warning(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info("\n✅ Test passed!")
        else:
            logger.error(f"\n❌ Test failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error running test: {e}")
        return False

def run_integration_tests():
    """Run only integration tests with real components."""
    logger.info("🧪 Running Integration Tests (Real Components)")
    logger.info("=" * 60)
    
    test_file = Path(__file__).parent / "test_master_graph_builder.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_file),
            "-m", "integration",
            "-v",
            "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Errors/Warnings:")
            logger.warning(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info("\n✅ All integration tests passed!")
        else:
            logger.error(f"\n❌ Integration tests failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error running integration tests: {e}")
        return False

def run_unit_tests():
    """Run only unit tests (mocked components)."""
    print("🧪 Running Unit Tests (Mocked Components)")
    print("=" * 60)
    
    test_file = Path(__file__).parent / "test_master_graph_builder.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_file),
            "-m", "not integration",
            "-v",
            "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Errors/Warnings:")
            logger.warning(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info("\n✅ All unit tests passed!")
        else:
            logger.error(f"\n❌ Unit tests failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error running unit tests: {e}")
        return False

def run_all_tests():
    """Run all tests (unit + integration)."""
    logger.info("🧪 Running All Tests (Unit + Integration)")
    logger.info("=" * 60)
    
    test_file = Path(__file__).parent / "test_master_graph_builder.py"
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            str(test_file),
            "-v",
            "--tb=short"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.stderr:
            logger.warning("Errors/Warnings:")
            logger.warning(result.stderr)
        
        success = result.returncode == 0
        
        if success:
            logger.info("\n✅ All tests passed!")
        else:
            logger.error(f"\n❌ Tests failed with return code: {result.returncode}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function to run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MasterGraphBuilder tests")
    parser.add_argument(
        "--test", "-t",
        help="Run a specific test (e.g., 'test_initialization')"
    )
    parser.add_argument(
        "--class", "-c",
        dest="test_class",
        help="Run a specific test class (e.g., 'TestMasterGraphBuilder')"
    )
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Run only integration tests with real components"
    )
    parser.add_argument(
        "--unit", "-u",
        action="store_true",
        help="Run only unit tests (mocked components)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all tests (unit + integration)"
    )
    
    args = parser.parse_args()
    
    if args.integration:
        success = run_integration_tests()
    elif args.unit:
        success = run_unit_tests()
    elif args.test:
        success = run_specific_test(args.test)
    elif args.test_class:
        success = run_specific_test(args.test_class)
    elif args.all:
        success = run_all_tests()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
