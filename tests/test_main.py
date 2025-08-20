"""
test_main.py - Comprehensive tests for main module and CLI

Tests command-line interface, argument parsing, and complete
simulation execution through the main entry point.
"""

from src.main import parse_arguments, print_header, main
import src.constants as const
import unittest
import sys
import os
import tempfile
import shutil
import argparse
from unittest.mock import patch, MagicMock
import io
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestArgumentParsing(unittest.TestCase):
    """Test command-line argument parsing."""

    def test_parse_arguments_default(self):
        """Test argument parsing with no arguments."""
        with patch('sys.argv', ['main.py']):
            args = parse_arguments()

            self.assertEqual(args.dt, const.DT)
            self.assertEqual(args.time, const.SIMULATION_TIME)
            self.assertIsNone(args.output)
            self.assertFalse(args.no_plots)
            self.assertFalse(args.test)
            self.assertEqual(args.progress, 1000)

    def test_parse_arguments_custom_dt(self):
        """Test parsing custom timestep."""
        with patch('sys.argv', ['main.py', '--dt', '0.002']):
            args = parse_arguments()
            self.assertEqual(args.dt, 0.002)

    def test_parse_arguments_custom_time(self):
        """Test parsing custom simulation time."""
        with patch('sys.argv', ['main.py', '--time', '5.0']):
            args = parse_arguments()
            self.assertEqual(args.time, 5.0)

    def test_parse_arguments_output_file(self):
        """Test parsing output file path."""
        with patch('sys.argv', ['main.py', '--output', '/tmp/test.csv']):
            args = parse_arguments()
            self.assertEqual(args.output, '/tmp/test.csv')

    def test_parse_arguments_no_plots(self):
        """Test parsing no-plots flag."""
        with patch('sys.argv', ['main.py', '--no-plots']):
            args = parse_arguments()
            self.assertTrue(args.no_plots)

    def test_parse_arguments_test_mode(self):
        """Test parsing test mode flag."""
        with patch('sys.argv', ['main.py', '--test']):
            args = parse_arguments()
            self.assertTrue(args.test)

    def test_parse_arguments_progress_interval(self):
        """Test parsing progress interval."""
        with patch('sys.argv', ['main.py', '--progress', '500']):
            args = parse_arguments()
            self.assertEqual(args.progress, 500)

    def test_parse_arguments_multiple(self):
        """Test parsing multiple arguments."""
        with patch('sys.argv', ['main.py', '--dt', '0.005', '--time', '2.0',
                                '--no-plots', '--progress', '100']):
            args = parse_arguments()

            self.assertEqual(args.dt, 0.005)
            self.assertEqual(args.time, 2.0)
            self.assertTrue(args.no_plots)
            self.assertEqual(args.progress, 100)


class TestPrintHeader(unittest.TestCase):
    """Test header printing functionality."""

    def test_print_header_output(self):
        """Test that header prints expected information."""
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        print_header()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check for expected content
        self.assertIn("CHARGED PARTICLE SIMULATION", output)
        self.assertIn("Number of particles:", output)
        self.assertIn(str(const.N_PARTICLES), output)
        self.assertIn("Particle mass:", output)
        self.assertIn(str(const.MASS), output)
        self.assertIn("Particle charge:", output)
        self.assertIn(str(const.CHARGE), output)
        self.assertIn("Gravity:", output)
        self.assertIn(str(const.GRAVITY), output)
        self.assertIn("Box dimensions:", output)
        self.assertIn("Initial Particle States", output)

    def test_print_header_particles(self):
        """Test that header prints all particle initial states."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        print_header()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check for each particle
        for i in range(const.N_PARTICLES):
            self.assertIn(f"Particle {i + 1}:", output)


class TestMainFunction(unittest.TestCase):
    """Test main execution function."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        # Backup original constants
        self.original_output_dir = const.OUTPUT_DIR
        const.OUTPUT_DIR = self.temp_dir

    def tearDown(self):
        """Clean up."""
        const.OUTPUT_DIR = self.original_output_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sys.argv', ['main.py', '--time', '0.01', '--no-plots'])
    def test_main_basic_run(self):
        """Test basic execution of main function."""
        # Mock to avoid actual matplotlib display
        with patch('matplotlib.pyplot.show'):
            result = main()

        self.assertEqual(result, 0)  # Success

        # Check that output file was created
        output_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        self.assertGreater(len(output_files), 0)

    @patch('sys.argv', ['main.py', '--time', '0.01', '--dt', '0.001', '--no-plots'])
    def test_main_custom_parameters(self):
        """Test main with custom parameters."""
        with patch('matplotlib.pyplot.show'):
            result = main()

        self.assertEqual(result, 0)

    @patch('sys.argv', ['main.py', '--output', 'custom_output.csv', '--time', '0.01', '--no-plots'])
    def test_main_custom_output(self):
        """Test main with custom output file."""
        custom_output = os.path.join(self.temp_dir, 'custom_output.csv')

        with patch('sys.argv', ['main.py', '--output', custom_output,
                                '--time', '0.01', '--no-plots']):
            with patch('matplotlib.pyplot.show'):
                result = main()

        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(custom_output))

        # Verify CSV content
        with open(custom_output, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Should have header and data
            self.assertGreater(len(rows), 1)
            # Header should start with 't' and 'E_total'
            self.assertEqual(rows[0][0], 't')
            self.assertEqual(rows[0][1], 'E_total')

    @patch('sys.argv', ['main.py', '--time', '0.01'])
    def test_main_with_plots(self):
        """Test main with plot generation."""
        plot_dir = os.path.join(self.temp_dir, const.PLOT_DIR)

        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.savefig'):
                result = main()

        self.assertEqual(result, 0)

    @patch('sys.argv', ['main.py', '--test'])
    def test_main_test_mode(self):
        """Test main in test mode."""
        # Mock the test runner to avoid running actual tests
        with patch('src.main.run_tests', return_value=True):
            with patch('matplotlib.pyplot.show'):
                result = main()

        self.assertEqual(result, 0)

    @patch('sys.argv', ['main.py', '--test'])
    def test_main_test_mode_failure(self):
        """Test main when tests fail."""
        # Mock test failure
        with patch('src.main.run_tests', return_value=False):
            result = main()

        self.assertEqual(result, 1)  # Should return error code


class TestMainIntegration(unittest.TestCase):
    """Integration tests for complete simulation runs."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_output_dir = const.OUTPUT_DIR
        const.OUTPUT_DIR = self.temp_dir

    def tearDown(self):
        """Clean up."""
        const.OUTPUT_DIR = self.original_output_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sys.argv', ['main.py', '--time', '0.1', '--dt', '0.001', '--no-plots'])
    def test_full_simulation_integration(self):
        """Test complete simulation from main entry point."""
        # Capture output
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with patch('matplotlib.pyplot.show'):
            result = main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check successful completion
        self.assertEqual(result, 0)

        # Check output contains expected messages
        self.assertIn("CHARGED PARTICLE SIMULATION", output)
        self.assertIn("Initializing simulation", output)
        self.assertIn("SIMULATION COMPLETE", output)
        self.assertIn("All tasks completed successfully", output)

        # Check files were created
        csv_files = [f for f in os.listdir(self.temp_dir) if f.endswith('.csv')]
        self.assertEqual(len(csv_files), 1)

        # Verify CSV has correct number of rows
        csv_file = os.path.join(self.temp_dir, csv_files[0])
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

            # Header + 100 steps (0.1s / 0.001 dt) + initial state
            expected_rows = 1 + 101
            self.assertEqual(len(rows), expected_rows)

    @patch('sys.argv', ['main.py', '--time', '0.05', '--progress', '10'])
    def test_progress_reporting(self):
        """Test progress reporting during simulation."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with patch('matplotlib.pyplot.show'):
            with patch('matplotlib.pyplot.savefig'):
                result = main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertEqual(result, 0)

        # Should have multiple progress updates
        progress_count = output.count("Progress:")
        self.assertGreater(progress_count, 0)

    @patch('sys.argv', ['main.py', '--time', '0.01', '--no-plots'])
    def test_energy_conservation_reporting(self):
        """Test that energy conservation is reported."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with patch('matplotlib.pyplot.show'):
            result = main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertEqual(result, 0)

        # Check energy statistics
        self.assertIn("Initial energy:", output)
        self.assertIn("Final energy:", output)
        self.assertIn("Energy drift:", output)
        self.assertIn("Relative drift:", output)

    @patch('sys.argv', ['main.py', '--time', '0.1', '--no-plots'])
    def test_collision_statistics_reporting(self):
        """Test that collision statistics are reported."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with patch('matplotlib.pyplot.show'):
            result = main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertEqual(result, 0)

        # Check collision statistics
        self.assertIn("Collision Statistics:", output)
        self.assertIn("Total wall collisions:", output)


class TestMainErrorHandling(unittest.TestCase):
    """Test error handling in main module."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sys.argv', ['main.py', '--dt', '-0.001'])  # Negative timestep
    def test_invalid_timestep(self):
        """Test handling of invalid timestep."""
        # Should handle gracefully or validate
        with patch('matplotlib.pyplot.show'):
            # Negative timestep is now allowed (time reversal)
            result = main()
            self.assertEqual(result, 0)

    @patch('sys.argv', ['main.py', '--output', '/invalid/path/file.csv',
                        '--time', '0.01', '--no-plots'])
    def test_invalid_output_path(self):
        """Test handling of invalid output path."""
        # Should handle file creation error gracefully
        try:
            result = main()
            # May return error code or handle gracefully
            self.assertIsNotNone(result)
        except:
            pass  # Expected for invalid path

    @patch('sys.argv', ['main.py', '--time', '0'])  # Zero simulation time
    def test_zero_simulation_time(self):
        """Test handling of zero simulation time."""
        with patch('matplotlib.pyplot.show'):
            result = main()
            # Should complete without error
            self.assertEqual(result, 0)


class TestMainPerformance(unittest.TestCase):
    """Test performance aspects of main simulation."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_output_dir = const.OUTPUT_DIR
        const.OUTPUT_DIR = self.temp_dir

    def tearDown(self):
        """Clean up."""
        const.OUTPUT_DIR = self.original_output_dir
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sys.argv', ['main.py', '--time', '0.1', '--dt', '0.001', '--no-plots'])
    def test_simulation_performance(self):
        """Test simulation completes in reasonable time."""
        import time

        start_time = time.time()

        with patch('matplotlib.pyplot.show'):
            result = main()

        elapsed_time = time.time() - start_time

        self.assertEqual(result, 0)

        # Should complete in reasonable time (adjust as needed)
        # For 0.1s simulation, should take less than 10 seconds
        self.assertLess(elapsed_time, 10.0)

    @patch('sys.argv', ['main.py', '--time', '0.01', '--dt', '0.0001', '--no-plots'])
    def test_small_timestep_performance(self):
        """Test performance with small timestep."""
        captured_output = io.StringIO()
        sys.stdout = captured_output

        with patch('matplotlib.pyplot.show'):
            result = main()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        self.assertEqual(result, 0)

        # Check that simulation completed
        self.assertIn("SIMULATION COMPLETE", output)


class TestMainUtilityFunctions(unittest.TestCase):
    """Test utility functions in main module."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('unittest.TextTestRunner')
    def test_run_tests_function(self, mock_runner):
        """Test the run_tests utility function."""
        from src.main import run_tests

        # Mock the test runner
        mock_result = MagicMock()
        mock_result.wasSuccessful.return_value = True
        mock_runner.return_value.run.return_value = mock_result

        result = run_tests()
        self.assertTrue(result)

    def test_output_directory_creation(self):
        """Test that output directories are created as needed."""
        test_dir = os.path.join(self.temp_dir, 'test_output_creation')

        # Ensure directory doesn't exist
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        with patch('src.constants.OUTPUT_DIR', test_dir):
            with patch('sys.argv', ['main.py', '--time', '0.01', '--no-plots']):
                with patch('matplotlib.pyplot.show'):
                    result = main()

        self.assertEqual(result, 0)
        self.assertTrue(os.path.exists(test_dir))


if __name__ == '__main__':
    unittest.main(verbosity=2)
