#!/usr/bin/env python3
"""
run_all_tests.py - Main test runner for all unit tests

This script discovers and runs all unit tests in the test suite,
providing comprehensive coverage reports and validation.
"""

import unittest
import sys
import os
import time
import argparse
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_tests(verbosity=2, pattern='test_*.py', failfast=False):
    """
    Run all unit tests and return results.

    Args:
        verbosity: Test output verbosity (0=quiet, 1=normal, 2=verbose)
        pattern: File pattern for test discovery
        failfast: Stop on first failure

    Returns:
        TestResult object
    """
    # Create test loader
    loader = unittest.TestLoader()

    # Discover tests
    test_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_dir, pattern=pattern)

    # Create test runner
    runner = unittest.TextTestRunner(
        verbosity=verbosity,
        failfast=failfast,
        stream=sys.stdout
    )

    # Run tests
    print("=" * 70)
    print("RUNNING CHARGED PARTICLE SIMULATION UNIT TESTS")
    print("=" * 70)
    print(f"Test directory: {test_dir}")
    print(f"Pattern: {pattern}")
    print()

    start_time = time.time()
    result = runner.run(suite)
    elapsed_time = time.time() - start_time

    return result, elapsed_time


def run_specific_test_module(module_name, verbosity=2):
    """
    Run tests from a specific module.

    Args:
        module_name: Name of test module (e.g., 'test_particle')
        verbosity: Test output verbosity

    Returns:
        TestResult object
    """
    loader = unittest.TestLoader()

    try:
        # Import the specific test module
        test_module = __import__(module_name)
        suite = loader.loadTestsFromModule(test_module)

        runner = unittest.TextTestRunner(verbosity=verbosity)
        print(f"\nRunning tests from {module_name}")
        print("-" * 50)

        result = runner.run(suite)
        return result

    except ImportError as e:
        print(f"Error: Could not import {module_name}: {e}")
        return None


def print_summary(result, elapsed_time):
    """
    Print test results summary.

    Args:
        result: TestResult object
        elapsed_time: Time taken to run tests
    """
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    # Basic statistics
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    success = total_tests - failures - errors - skipped

    print(f"Tests run:     {total_tests}")
    print(f"Successes:     {success}")
    print(f"Failures:      {failures}")
    print(f"Errors:        {errors}")
    print(f"Skipped:       {skipped}")
    print(f"Time elapsed:  {elapsed_time:.2f} seconds")

    # Success rate
    if total_tests > 0:
        success_rate = (success / total_tests) * 100
        print(f"Success rate:  {success_rate:.1f}%")

    print("=" * 70)

    # Detailed failure information
    if failures:
        print("\nFAILURES:")
        print("-" * 40)
        for test, traceback in result.failures:
            print(f"\n{test}:")
            print(traceback)

    if errors:
        print("\nERRORS:")
        print("-" * 40)
        for test, traceback in result.errors:
            print(f"\n{test}:")
            print(traceback)

    # Final status
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 70)


def run_coverage_analysis():
    """
    Run tests with code coverage analysis.

    Requires: pip install coverage
    """
    try:
        import coverage

        print("Running tests with coverage analysis...")
        print("-" * 50)

        # Start coverage
        cov = coverage.Coverage(source=['src'])
        cov.start()

        # Run tests
        result, elapsed = run_tests(verbosity=1)

        # Stop coverage
        cov.stop()
        cov.save()

        # Print coverage report
        print("\n" + "=" * 70)
        print("COVERAGE REPORT")
        print("=" * 70)

        # Create string buffer for report
        buffer = StringIO()
        cov.report(file=buffer)
        print(buffer.getvalue())

        # Generate HTML report
        print("\nGenerating HTML coverage report...")
        cov.html_report(directory='htmlcov')
        print("HTML report saved to: htmlcov/index.html")

        return result

    except ImportError:
        print("Coverage module not installed.")
        print("Install with: pip install coverage")
        print("Running tests without coverage...")
        result, elapsed = run_tests()
        return result


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(
        description='Run unit tests for charged particle simulation'
    )

    parser.add_argument(
        '-v', '--verbosity',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test output verbosity (0=quiet, 1=normal, 2=verbose)'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='test_*.py',
        help='File pattern for test discovery'
    )

    parser.add_argument(
        '-f', '--failfast',
        action='store_true',
        help='Stop on first test failure'
    )

    parser.add_argument(
        '-m', '--module',
        help='Run specific test module (e.g., test_particle)'
    )

    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Run with code coverage analysis'
    )

    parser.add_argument(
        '--critical-only',
        action='store_true',
        help='Run only critical tests (forces, integrator, box)'
    )

    args = parser.parse_args()

    # Run coverage analysis if requested
    if args.coverage:
        result = run_coverage_analysis()
        return 0 if result.wasSuccessful() else 1

    # Run specific module if requested
    if args.module:
        result = run_specific_test_module(args.module, args.verbosity)
        return 0 if result and result.wasSuccessful() else 1

    # Run critical tests only if requested
    if args.critical_only:
        print("Running critical tests only...")
        critical_modules = [
            'test_forces',
            'test_integrator',
            'test_box',
            'test_regularization'
        ]

        all_passed = True
        for module in critical_modules:
            result = run_specific_test_module(module, args.verbosity)
            if result and not result.wasSuccessful():
                all_passed = False

        return 0 if all_passed else 1

    # Run all tests
    result, elapsed_time = run_tests(
        verbosity=args.verbosity,
        pattern=args.pattern,
        failfast=args.failfast
    )

    # Print summary
    print_summary(result, elapsed_time)

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(main())
