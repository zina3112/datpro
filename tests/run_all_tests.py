#!/usr/bin/env python3
"""
run_all_tests.py - Comprehensive test suite runner for charged particle simulation

This script runs all unit tests, integration tests, and generates a detailed
test report including coverage analysis and performance metrics.

Usage:
    python run_all_tests.py [options]

Options:
    --verbose, -v     Increase output verbosity
    --quiet, -q       Minimal output
    --failfast, -f    Stop on first failure
    --pattern PATTERN Test file pattern (default: test_*.py)
    --coverage        Generate coverage report
    --performance     Run performance tests
    --report FILE     Save report to file
"""

import unittest
import sys
import os
import time
import argparse
import json
from datetime import datetime
import traceback
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ColoredTextTestResult(unittest.TextTestResult):
    """Custom test result class with colored output."""

    # ANSI color codes
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.test_times = {}
        self.test_start_time = None

    def startTest(self, test):
        """Called when a test starts."""
        super().startTest(test)
        self.test_start_time = time.time()

    def addSuccess(self, test):
        """Called when a test passes."""
        super().addSuccess(test)
        elapsed = time.time() - self.test_start_time
        self.test_times[str(test)] = elapsed
        if self.showAll:
            self.stream.writeln(f"{self.GREEN}✓ OK{self.ENDC} ({elapsed:.3f}s)")
        elif self.dots:
            self.stream.write(f"{self.GREEN}.{self.ENDC}")
            self.stream.flush()

    def addError(self, test, err):
        """Called when a test raises an error."""
        super().addError(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ ERROR{self.ENDC}")
        elif self.dots:
            self.stream.write(f"{self.RED}E{self.ENDC}")
            self.stream.flush()

    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        if self.showAll:
            self.stream.writeln(f"{self.RED}✗ FAIL{self.ENDC}")
        elif self.dots:
            self.stream.write(f"{self.RED}F{self.ENDC}")
            self.stream.flush()

    def addSkip(self, test, reason):
        """Called when a test is skipped."""
        super().addSkip(test, reason)
        if self.showAll:
            self.stream.writeln(f"{self.YELLOW}⊘ SKIP{self.ENDC}: {reason}")
        elif self.dots:
            self.stream.write(f"{self.YELLOW}s{self.ENDC}")
            self.stream.flush()


class ColoredTextTestRunner(unittest.TextTestRunner):
    """Custom test runner with colored output."""

    resultclass = ColoredTextTestResult

    def run(self, test):
        """Run the test suite."""
        result = super().run(test)

        # Print summary with colors
        if result.wasSuccessful():
            print(f"\n{ColoredTextTestResult.GREEN}{ColoredTextTestResult.BOLD}"
                  f"ALL TESTS PASSED!{ColoredTextTestResult.ENDC}")
        else:
            print(f"\n{ColoredTextTestResult.RED}{ColoredTextTestResult.BOLD}"
                  f"TESTS FAILED!{ColoredTextTestResult.ENDC}")

        return result


class TestSuiteRunner:
    """Main test suite runner with advanced features."""

    def __init__(self, args):
        """Initialize the test runner."""
        self.args = args
        self.results = {}
        self.start_time = None
        self.end_time = None

    def discover_tests(self):
        """Discover all test modules."""
        loader = unittest.TestLoader()
        test_dir = os.path.dirname(os.path.abspath(__file__))

        # Discover tests
        suite = loader.discover(
            test_dir,
            pattern=self.args.pattern,
            top_level_dir=os.path.dirname(test_dir)
        )

        return suite

    def count_tests(self, suite):
        """Count total number of tests in suite."""
        count = 0
        for test_group in suite:
            if hasattr(test_group, '__iter__'):
                for test in test_group:
                    if hasattr(test, '__iter__'):
                        count += len(list(test))
                    else:
                        count += 1
            else:
                count += 1
        return count

    def run_tests(self):
        """Run all tests and collect results."""
        print("=" * 70)
        print("CHARGED PARTICLE SIMULATION - COMPREHENSIVE TEST SUITE")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # Discover tests
        print("Discovering tests...")
        suite = self.discover_tests()
        test_count = self.count_tests(suite)
        print(f"Found {test_count} tests")
        print()

        # Configure runner
        if self.args.verbose:
            verbosity = 2
        elif self.args.quiet:
            verbosity = 0
        else:
            verbosity = 1

        # Use colored runner if not quiet
        if not self.args.quiet:
            runner = ColoredTextTestRunner(
                verbosity=verbosity,
                failfast=self.args.failfast
            )
        else:
            runner = unittest.TextTestRunner(
                verbosity=verbosity,
                failfast=self.args.failfast
            )

        # Run tests
        print("Running tests...")
        print("-" * 70)
        self.start_time = time.time()

        result = runner.run(suite)

        self.end_time = time.time()
        print("-" * 70)

        # Store results
        self.results = {
            'total': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'duration': self.end_time - self.start_time
        }

        # Add detailed results if verbose
        if self.args.verbose:
            self.results['failures_detail'] = [
                (str(test), traceback.format_exception(*err))
                for test, err in result.failures
            ]
            self.results['errors_detail'] = [
                (str(test), traceback.format_exception(*err))
                for test, err in result.errors
            ]

        return result

    def run_performance_tests(self):
        """Run performance-specific tests."""
        if not self.args.performance:
            return

        print("\n" + "=" * 70)
        print("PERFORMANCE TESTS")
        print("=" * 70)

        perf_results = {}

        # Test 1: Simulation speed
        print("\n1. Testing simulation speed...")
        from src.simulation import Simulation
        import tempfile

        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            sim = Simulation(dt=0.001, output_file=tmp.name)

            start = time.time()
            sim.run(simulation_time=1.0, progress_interval=1000)
            elapsed = time.time() - start

            steps_per_second = 1000 / elapsed
            perf_results['simulation_speed'] = {
                'steps_per_second': steps_per_second,
                'time_for_1s_simulation': elapsed,
                'realtime_factor': 1.0 / elapsed
            }

            print(f"  - Steps per second: {steps_per_second:.0f}")
            print(f"  - Time for 1s simulation: {elapsed:.3f}s")
            print(f"  - Realtime factor: {1.0/elapsed:.2f}x")

        # Test 2: Memory usage
        print("\n2. Testing memory usage...")
        import tracemalloc

        tracemalloc.start()

        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            sim = Simulation(dt=0.001, output_file=tmp.name)
            sim.run(simulation_time=0.1, progress_interval=1000)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            perf_results['memory_usage'] = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }

            print(f"  - Current memory: {current/1024/1024:.2f} MB")
            print(f"  - Peak memory: {peak/1024/1024:.2f} MB")

        self.results['performance'] = perf_results

    def generate_coverage_report(self):
        """Generate code coverage report."""
        if not self.args.coverage:
            return

        print("\n" + "=" * 70)
        print("CODE COVERAGE ANALYSIS")
        print("=" * 70)

        try:
            import coverage

            # Start coverage
            cov = coverage.Coverage(source=['src'])
            cov.start()

            # Run tests again with coverage
            suite = self.discover_tests()
            runner = unittest.TextTestRunner(verbosity=0)
            runner.run(suite)

            # Stop coverage
            cov.stop()
            cov.save()

            # Generate report
            print("\nCoverage Report:")
            print("-" * 50)

            # Capture report
            report_stream = StringIO()
            cov.report(file=report_stream)
            report = report_stream.getvalue()
            print(report)

            # Parse coverage percentage
            lines = report.split('\n')
            for line in lines:
                if 'TOTAL' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        coverage_pct = parts[-1].rstrip('%')
                        self.results['coverage'] = float(coverage_pct)

        except ImportError:
            print("Coverage module not installed. Install with: pip install coverage")

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)

        total = self.results['total']
        passed = total - self.results['failures'] - self.results['errors']

        print(f"Tests run:     {total}")
        print(f"Passed:        {passed} ({passed/total*100:.1f}%)")
        print(f"Failed:        {self.results['failures']}")
        print(f"Errors:        {self.results['errors']}")
        print(f"Skipped:       {self.results['skipped']}")
        print(f"Duration:      {self.results['duration']:.3f} seconds")

        if 'coverage' in self.results:
            print(f"Code coverage: {self.results['coverage']:.1f}%")

        if 'performance' in self.results:
            perf = self.results['performance']
            if 'simulation_speed' in perf:
                print(f"\nPerformance:")
                print(f"  Simulation speed: {perf['simulation_speed']['steps_per_second']:.0f} steps/s")
                print(f"  Memory usage: {perf['memory_usage']['peak_mb']:.2f} MB")

        print("\n" + "=" * 70)

        if self.results['success']:
            print(f"{ColoredTextTestResult.GREEN}{ColoredTextTestResult.BOLD}"
                  f"✓ ALL TESTS PASSED{ColoredTextTestResult.ENDC}")
        else:
            print(f"{ColoredTextTestResult.RED}{ColoredTextTestResult.BOLD}"
                  f"✗ TESTS FAILED{ColoredTextTestResult.ENDC}")

        print("=" * 70)

    def save_report(self):
        """Save test report to file."""
        if not self.args.report:
            return

        report_data = {
            'timestamp': datetime.now().isoformat(),
            'results': self.results,
            'arguments': vars(self.args)
        }

        with open(self.args.report, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        print(f"\nReport saved to: {self.args.report}")

    def run(self):
        """Run the complete test suite."""
        try:
            # Run main tests
            result = self.run_tests()

            # Run performance tests if requested
            self.run_performance_tests()

            # Generate coverage report if requested
            self.generate_coverage_report()

            # Print summary
            self.print_summary()

            # Save report if requested
            self.save_report()

            # Return exit code
            return 0 if result.wasSuccessful() else 1

        except Exception as e:
            print(f"\nError running tests: {e}")
            traceback.print_exc()
            return 2


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive test suite for charged particle simulation'
    )

    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Increase output verbosity'
    )

    parser.add_argument(
        '--quiet', '-q', action='store_true',
        help='Minimal output'
    )

    parser.add_argument(
        '--failfast', '-f', action='store_true',
        help='Stop on first failure'
    )

    parser.add_argument(
        '--pattern', default='test_*.py',
        help='Test file pattern (default: test_*.py)'
    )

    parser.add_argument(
        '--coverage', action='store_true',
        help='Generate coverage report'
    )

    parser.add_argument(
        '--performance', action='store_true',
        help='Run performance tests'
    )

    parser.add_argument(
        '--report', type=str,
        help='Save report to file'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Handle matplotlib backend for tests
    import matplotlib
    matplotlib.use('Agg')

    # Create and run test suite
    runner = TestSuiteRunner(args)
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
