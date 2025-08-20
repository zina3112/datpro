"""
test_data_handler.py - Comprehensive unit tests for DataHandler class

Tests all data recording, file I/O, trajectory management, and
statistics calculation functionality.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import csv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_handler import DataHandler
from src.particle import Particle
import src.constants as const


class TestDataHandlerInitialization(unittest.TestCase):
    """Test DataHandler initialization."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_default(self):
        """Test initialization with default output file."""
        # Temporarily change output directory
        original_dir = const.OUTPUT_DIR
        const.OUTPUT_DIR = self.temp_dir

        try:
            handler = DataHandler()

            self.assertIsNotNone(handler.output_file)
            self.assertIn(self.temp_dir, handler.output_file)
            self.assertEqual(len(handler.data_buffer), 0)
            self.assertEqual(handler.records_written, 0)

        finally:
            const.OUTPUT_DIR = original_dir

    def test_initialization_custom_file(self):
        """Test initialization with custom output file."""
        output_file = os.path.join(self.temp_dir, "custom.csv")
        handler = DataHandler(output_file)

        self.assertEqual(handler.output_file, output_file)
        self.assertEqual(len(handler.data_buffer), 0)

    def test_header_generation(self):
        """Test CSV header generation."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        expected_columns = 2 + 4 * const.N_PARTICLES  # t, E_total, then x,y,vx,vy for each
        self.assertEqual(len(handler.header), expected_columns)

        # Check header format
        self.assertEqual(handler.header[0], 't')
        self.assertEqual(handler.header[1], 'E_total')
        self.assertEqual(handler.header[2], 'x1')
        self.assertEqual(handler.header[3], 'y1')
        self.assertEqual(handler.header[4], 'vx1')
        self.assertEqual(handler.header[5], 'vy1')

    def test_trajectory_data_initialization(self):
        """Test trajectory data structure initialization."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        self.assertIn('time', handler.trajectory_data)
        self.assertIn('energy', handler.trajectory_data)
        self.assertIn('particles', handler.trajectory_data)

        self.assertEqual(len(handler.trajectory_data['time']), 0)
        self.assertEqual(len(handler.trajectory_data['energy']), 0)
        self.assertEqual(len(handler.trajectory_data['particles']), 0)


class TestDataRecording(unittest.TestCase):
    """Test data recording functionality."""

    def setUp(self):
        """Create temporary directory and test particles."""
        self.temp_dir = tempfile.mkdtemp()
        self.particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0)
        ]

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_record_state_basic(self):
        """Test basic state recording."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        time = 1.5
        energy = 1234.5

        handler.record_state(time, energy, self.particles)

        self.assertEqual(len(handler.data_buffer), 1)
        self.assertEqual(handler.records_written, 1)

        # Check recorded data
        row = handler.data_buffer[0]
        self.assertEqual(row[0], time)
        self.assertEqual(row[1], energy)
        self.assertEqual(row[2], 10.0)  # x1
        self.assertEqual(row[3], 20.0)  # y1
        self.assertEqual(row[4], 5.0)   # vx1
        self.assertEqual(row[5], -3.0)  # vy1
        self.assertEqual(row[6], 30.0)  # x2
        self.assertEqual(row[7], 40.0)  # y2

    def test_record_state_multiple(self):
        """Test recording multiple states."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        for i in range(5):
            handler.record_state(float(i), 1000.0 + i, self.particles)

        self.assertEqual(len(handler.data_buffer), 5)
        self.assertEqual(handler.records_written, 5)

        # Check times are correct
        for i, row in enumerate(handler.data_buffer):
            self.assertEqual(row[0], float(i))

    def test_trajectory_data_storage(self):
        """Test that trajectory data is stored correctly."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        handler.record_state(1.0, 100.0, self.particles)
        handler.record_state(2.0, 101.0, self.particles)

        self.assertEqual(handler.trajectory_data['time'], [1.0, 2.0])
        self.assertEqual(handler.trajectory_data['energy'], [100.0, 101.0])

        # Check particle data
        self.assertEqual(len(handler.trajectory_data['particles']), 2)
        first_timestep = handler.trajectory_data['particles'][0]
        self.assertEqual(first_timestep[0]['x'], 10.0)
        self.assertEqual(first_timestep[0]['y'], 20.0)

    def test_record_state_with_many_particles(self):
        """Test recording with many particles."""
        # Create 10 particles
        particles = [
            Particle(x=float(i), y=float(i*2), vx=1.0, vy=-1.0)
            for i in range(10)
        ]

        # Need to adjust N_PARTICLES temporarily
        original_n = const.N_PARTICLES
        const.N_PARTICLES = 10

        try:
            handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))
            handler.record_state(0.0, 500.0, particles)

            row = handler.data_buffer[0]
            # Should have t, E, then 4 values per particle
            self.assertEqual(len(row), 2 + 4 * 10)

        finally:
            const.N_PARTICLES = original_n


class TestFileSaving(unittest.TestCase):
    """Test file saving functionality."""

    def setUp(self):
        """Create temporary directory and test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0)
        ]

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_save_basic(self):
        """Test basic file saving."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        handler = DataHandler(output_file)

        # Record some data
        for i in range(3):
            handler.record_state(float(i), 100.0 + i, self.particles)

        # Save to file
        handler.save()

        # Check file exists
        self.assertTrue(os.path.exists(output_file))

        # Read and verify content
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + 3 data rows
        self.assertEqual(len(rows), 4)

        # Check header
        self.assertEqual(rows[0][0], 't')
        self.assertEqual(rows[0][1], 'E_total')

        # Check first data row
        self.assertEqual(float(rows[1][0]), 0.0)
        self.assertEqual(float(rows[1][1]), 100.0)

    def test_save_empty(self):
        """Test saving with no data."""
        output_file = os.path.join(self.temp_dir, "empty.csv")
        handler = DataHandler(output_file)

        handler.save()

        # File should exist with just header
        self.assertTrue(os.path.exists(output_file))

        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1)  # Just header

    def test_save_alternative_filename(self):
        """Test saving to alternative filename."""
        handler = DataHandler(os.path.join(self.temp_dir, "original.csv"))

        handler.record_state(0.0, 100.0, self.particles)

        alt_file = os.path.join(self.temp_dir, "alternative.csv")
        handler.save(alt_file)

        self.assertTrue(os.path.exists(alt_file))

    def test_save_incremental(self):
        """Test incremental saving (append mode)."""
        output_file = os.path.join(self.temp_dir, "incremental.csv")
        handler = DataHandler(output_file)

        # First save
        handler.record_state(0.0, 100.0, self.particles)
        handler.save_incremental()

        # Record more data
        handler.record_state(1.0, 101.0, self.particles)
        handler.save_incremental()

        # Read file
        with open(output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Should have header + 2 data rows
        self.assertEqual(len(rows), 3)

    def test_save_error_handling(self):
        """Test error handling in save."""
        # Try to save to invalid path
        handler = DataHandler("/invalid/path/test.csv")
        handler.record_state(0.0, 100.0, self.particles)

        # Should handle error gracefully (print message, not crash)
        try:
            handler.save()
        except:
            self.fail("Save should handle errors gracefully")


class TestTrajectoryRetrieval(unittest.TestCase):
    """Test trajectory retrieval methods."""

    def setUp(self):
        """Create test handler with data."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create test particles and record multiple states
        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0)
        ]

        # Simulate motion
        for i in range(5):
            # Update positions for simulation
            particles[0].state[0] = 10.0 + i * 5.0  # x increases
            particles[0].state[1] = 20.0 - i * 3.0  # y decreases
            particles[1].state[0] = 30.0 - i * 5.0  # x decreases
            particles[1].state[1] = 40.0 + i * 3.0  # y increases

            self.handler.record_state(float(i), 100.0 + i, particles)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_particle_trajectory(self):
        """Test getting single particle trajectory."""
        x_pos, y_pos = self.handler.get_particle_trajectory(0)

        self.assertEqual(len(x_pos), 5)
        self.assertEqual(len(y_pos), 5)

        # Check trajectory is correct
        for i in range(5):
            self.assertEqual(x_pos[i], 10.0 + i * 5.0)
            self.assertEqual(y_pos[i], 20.0 - i * 3.0)

    def test_get_particle_trajectory_invalid_index(self):
        """Test getting trajectory with invalid index."""
        with self.assertRaises(ValueError):
            self.handler.get_particle_trajectory(10)  # Only 2 particles

    def test_get_all_trajectories(self):
        """Test getting all particle trajectories."""
        # Need to set N_PARTICLES correctly
        original_n = const.N_PARTICLES
        const.N_PARTICLES = 2

        try:
            trajectories = self.handler.get_all_trajectories()

            self.assertEqual(len(trajectories), 2)

            # Check first particle
            x0, y0 = trajectories[0]
            self.assertEqual(len(x0), 5)
            self.assertEqual(x0[0], 10.0)
            self.assertEqual(x0[-1], 30.0)

            # Check second particle
            x1, y1 = trajectories[1]
            self.assertEqual(len(x1), 5)
            self.assertEqual(x1[0], 30.0)
            self.assertEqual(x1[-1], 10.0)

        finally:
            const.N_PARTICLES = original_n

    def test_get_energy_history(self):
        """Test getting energy history."""
        times, energies = self.handler.get_energy_history()

        self.assertEqual(len(times), 5)
        self.assertEqual(len(energies), 5)

        for i in range(5):
            self.assertEqual(times[i], float(i))
            self.assertEqual(energies[i], 100.0 + i)


class TestDataLoading(unittest.TestCase):
    """Test loading data from files."""

    def setUp(self):
        """Create temporary directory and test file."""
        self.temp_dir = tempfile.mkdtemp()

        # Create a test CSV file
        self.test_file = os.path.join(self.temp_dir, "test_data.csv")

        # Need to set N_PARTICLES
        self.original_n = const.N_PARTICLES
        const.N_PARTICLES = 2

        # Create handler and save some data
        handler = DataHandler(self.test_file)
        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0)
        ]

        for i in range(3):
            handler.record_state(float(i), 100.0 + i, particles)

        handler.save()

    def tearDown(self):
        """Clean up temporary files."""
        const.N_PARTICLES = self.original_n
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_load_from_file(self):
        """Test loading data from CSV file."""
        new_handler = DataHandler(os.path.join(self.temp_dir, "dummy.csv"))
        new_handler.load_from_file(self.test_file)

        # Check data was loaded
        self.assertEqual(len(new_handler.trajectory_data['time']), 3)
        self.assertEqual(new_handler.trajectory_data['time'], [0.0, 1.0, 2.0])
        self.assertEqual(new_handler.trajectory_data['energy'], [100.0, 101.0, 102.0])

        # Check particle data
        first_particle = new_handler.trajectory_data['particles'][0][0]
        self.assertEqual(first_particle['x'], 10.0)
        self.assertEqual(first_particle['y'], 20.0)
        self.assertEqual(first_particle['vx'], 5.0)
        self.assertEqual(first_particle['vy'], -3.0)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        handler = DataHandler(os.path.join(self.temp_dir, "dummy.csv"))

        # Should handle error gracefully
        try:
            handler.load_from_file("/nonexistent/file.csv")
        except:
            self.fail("Should handle missing file gracefully")

    def test_load_empty_file(self):
        """Test loading empty CSV file."""
        empty_file = os.path.join(self.temp_dir, "empty.csv")

        # Create empty file with just header
        handler = DataHandler(empty_file)
        handler.save()  # Saves just header

        new_handler = DataHandler(os.path.join(self.temp_dir, "dummy.csv"))
        new_handler.load_from_file(empty_file)

        # Should have empty data
        self.assertEqual(len(new_handler.trajectory_data['time']), 0)


class TestStatistics(unittest.TestCase):
    """Test statistics calculation."""

    def setUp(self):
        """Create handler with test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create test data with known statistics
        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0)
        ]

        # Record data with varying energy
        energies = [100.0, 101.0, 99.0, 102.0, 98.0]
        for i, energy in enumerate(energies):
            self.handler.record_state(float(i), energy, particles)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_statistics_basic(self):
        """Test basic statistics calculation."""
        stats = self.handler.get_statistics()

        self.assertIn('initial_energy', stats)
        self.assertIn('final_energy', stats)
        self.assertIn('mean_energy', stats)
        self.assertIn('std_energy', stats)
        self.assertIn('max_energy', stats)
        self.assertIn('min_energy', stats)
        self.assertIn('energy_drift', stats)
        self.assertIn('relative_drift', stats)
        self.assertIn('num_timesteps', stats)

    def test_get_statistics_values(self):
        """Test that statistics are calculated correctly."""
        stats = self.handler.get_statistics()

        self.assertEqual(stats['initial_energy'], 100.0)
        self.assertEqual(stats['final_energy'], 98.0)
        self.assertEqual(stats['energy_drift'], -2.0)
        self.assertEqual(stats['relative_drift'], -0.02)
        self.assertEqual(stats['max_energy'], 102.0)
        self.assertEqual(stats['min_energy'], 98.0)
        self.assertEqual(stats['num_timesteps'], 5)

        # Check mean (should be 100.0)
        self.assertAlmostEqual(stats['mean_energy'], 100.0)

    def test_get_statistics_empty(self):
        """Test statistics with no data."""
        empty_handler = DataHandler(os.path.join(self.temp_dir, "empty.csv"))
        stats = empty_handler.get_statistics()

        # Should return empty dict
        self.assertEqual(stats, {})

    def test_get_statistics_single_point(self):
        """Test statistics with single data point."""
        handler = DataHandler(os.path.join(self.temp_dir, "single.csv"))
        particles = [Particle(x=0, y=0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)

        stats = handler.get_statistics()

        self.assertEqual(stats['initial_energy'], 100.0)
        self.assertEqual(stats['final_energy'], 100.0)
        self.assertEqual(stats['energy_drift'], 0.0)
        self.assertEqual(stats['relative_drift'], 0.0)
        self.assertEqual(stats['num_timesteps'], 1)


class TestDataHandlerEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_zero_particles(self):
        """Test handling zero particles."""
        original_n = const.N_PARTICLES
        const.N_PARTICLES = 0

        try:
            handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

            # Header should just have t and E_total
            self.assertEqual(len(handler.header), 2)

            # Should be able to record with empty particle list
            handler.record_state(0.0, 100.0, [])

            row = handler.data_buffer[0]
            self.assertEqual(len(row), 2)
            self.assertEqual(row[0], 0.0)
            self.assertEqual(row[1], 100.0)

        finally:
            const.N_PARTICLES = original_n

    def test_very_long_filename(self):
        """Test handling very long filenames."""
        long_name = "a" * 200 + ".csv"
        long_path = os.path.join(self.temp_dir, long_name)

        handler = DataHandler(long_path)
        particles = [Particle(x=0, y=0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)

        handler.save()

        # Should handle long filename
        self.assertTrue(os.path.exists(long_path))

    def test_special_characters_in_path(self):
        """Test handling special characters in path."""
        special_dir = os.path.join(self.temp_dir, "test dir with spaces")
        os.makedirs(special_dir, exist_ok=True)

        output_file = os.path.join(special_dir, "test file.csv")
        handler = DataHandler(output_file)

        particles = [Particle(x=0, y=0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)
        handler.save()

        self.assertTrue(os.path.exists(output_file))

    def test_very_large_dataset(self):
        """Test handling large amount of data."""
        handler = DataHandler(os.path.join(self.temp_dir, "large.csv"))
        particles = [Particle(x=0, y=0, vx=0, vy=0)]

        # Record 1000 timesteps
        for i in range(1000):
            handler.record_state(float(i), 100.0, particles)

        self.assertEqual(handler.records_written, 1000)
        self.assertEqual(len(handler.data_buffer), 1000)

        # Should be able to save
        handler.save()

        # Check file was created and has correct size
        self.assertTrue(os.path.exists(handler.output_file))

        with open(handler.output_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 1001)  # Header + 1000 data rows

    def test_nan_values(self):
        """Test handling NaN values."""
        handler = DataHandler(os.path.join(self.temp_dir, "nan.csv"))

        particles = [Particle(x=float('nan'), y=0, vx=0, vy=0)]

        # Should handle NaN without crashing
        handler.record_state(0.0, float('nan'), particles)
        handler.save()

        # File should exist
        self.assertTrue(os.path.exists(handler.output_file))

    def test_infinite_values(self):
        """Test handling infinite values."""
        handler = DataHandler(os.path.join(self.temp_dir, "inf.csv"))

        particles = [Particle(x=float('inf'), y=float('-inf'), vx=0, vy=0)]

        # Should handle infinity without crashing
        handler.record_state(0.0, float('inf'), particles)
        handler.save()

        # File should exist
        self.assertTrue(os.path.exists(handler.output_file))


if __name__ == '__main__':
    unittest.main(verbosity=2)
