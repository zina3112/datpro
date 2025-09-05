"""
test_data_handler.py - Unit tests for data recording and I/O

Tests CSV output, trajectory storage, and data retrieval with proper
particle count and data structure expectations.
"""

import src.constants as const
from src.particle import Particle
from src.data_handler import DataHandler
import unittest
import numpy as np
import sys
import os
import tempfile
import csv
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDataHandler(unittest.TestCase):
    """Test suite for DataHandler class."""

    def setUp(self):
        """Set up test data handler with correct particle count."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.test_dir, "test_data.csv")

        self.data_handler = DataHandler(self.test_file)

        # Create test particles matching system expectations (const.N_PARTICLES)
        self.particles = []
        for i in range(const.N_PARTICLES):
            x = 10.0 + i * 10.0
            y = 20.0 + i * 5.0
            vx = 5.0 - i * 1.0
            vy = -3.0 + i * 0.5
            particle = Particle(x=x, y=y, vx=vx, vy=vy)
            self.particles.append(particle)

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test data handler initialization."""
        self.assertEqual(self.data_handler.output_file, self.test_file)
        self.assertEqual(self.data_handler.records_written, 0)
        self.assertEqual(len(self.data_handler.trajectory_data['time']), 0)

    def test_header_generation(self):
        """Test CSV header generation matches specification."""
        expected_header = ['t', 'E_total']
        for i in range(const.N_PARTICLES):
            expected_header.extend(
                [f'x{i + 1}', f'y{i + 1}', f'vx{i + 1}', f'vy{i + 1}'])

        # Verify header structure
        self.assertEqual(self.data_handler.header, expected_header)
        self.assertEqual(len(self.data_handler.header),
                         2 + 4 * const.N_PARTICLES)

        # Verify specific columns exist
        self.assertEqual(self.data_handler.header[:2], ['t', 'E_total'])
        self.assertIn('x1', self.data_handler.header)
        self.assertIn('vy7', self.data_handler.header)

    def test_record_state(self):
        """Test recording simulation state."""
        time = 1.5
        energy = 100.5

        self.data_handler.record_state(time, energy, self.particles)

        # Verify recording
        self.assertEqual(len(self.data_handler.data_buffer), 1)
        self.assertEqual(self.data_handler.records_written, 1)

        # Check time and energy
        self.assertEqual(self.data_handler.trajectory_data['time'][0], time)
        self.assertEqual(
            self.data_handler.trajectory_data['energy'][0], energy)

        # Check particle data structure
        particle_data = self.data_handler.trajectory_data['particles'][0]
        self.assertEqual(len(particle_data), const.N_PARTICLES)

        # Verify specific particle values
        self.assertEqual(particle_data[0]['x'], 10.0)
        self.assertEqual(particle_data[0]['y'], 20.0)
        self.assertEqual(particle_data[1]['vx'], 4.0)
        self.assertEqual(particle_data[6]['x'], 70.0)

    def test_save_to_csv(self):
        """Test saving complete data to CSV file."""
        # Record data for multiple timesteps
        for t in range(3):
            self.data_handler.record_state(
                time=t * 0.1,
                total_energy=100.0 - t,
                particles=self.particles
            )

        # Save to file
        self.data_handler.save()

        # Verify file exists
        self.assertTrue(os.path.exists(self.test_file))

        # Read and verify CSV structure
        with open(self.test_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Check structure matches specification
        self.assertEqual(len(rows), 4)  # Header + 3 data rows
        self.assertEqual(len(rows[0]), 2 + 4 * const.N_PARTICLES)

        # Verify header and data
        self.assertEqual(rows[0][0], 't')
        self.assertEqual(rows[0][1], 'E_total')
        self.assertEqual(float(rows[1][0]), 0.0)
        self.assertEqual(float(rows[2][0]), 0.1)
        self.assertEqual(float(rows[3][1]), 98.0)

    def test_save_incremental(self):
        """Test incremental data saving."""
        # Record and save incrementally
        self.data_handler.record_state(0.0, 100.0, self.particles)
        self.data_handler.save_incremental()

        # Buffer should clear after save
        self.assertEqual(len(self.data_handler.data_buffer), 0)

        # Record more data
        self.data_handler.record_state(0.1, 99.0, self.particles)
        self.data_handler.save_incremental()

        # File should contain both records
        with open(self.test_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)

        self.assertEqual(len(rows), 3)  # Header + 2 data rows

    def test_get_particle_trajectory(self):
        """Test retrieving individual particle trajectory."""
        # Record timesteps with changing positions
        for t in range(5):
            for i, particle in enumerate(self.particles):
                particle.position = np.array([
                    10.0 + i * 10.0 + t,
                    20.0 + i * 5.0 + t * 0.5
                ])

            self.data_handler.record_state(t * 0.1, 100.0, self.particles)

        # Test first particle trajectory
        x_pos, y_pos = self.data_handler.get_particle_trajectory(0)

        self.assertEqual(len(x_pos), 5)
        self.assertEqual(len(y_pos), 5)

        # Verify trajectory values
        self.assertEqual(x_pos[0], 10.0)
        self.assertEqual(x_pos[4], 14.0)
        self.assertEqual(y_pos[0], 20.0)
        self.assertEqual(y_pos[4], 22.0)

        # Test last particle trajectory
        x_pos_last, y_pos_last = self.data_handler.get_particle_trajectory(
            const.N_PARTICLES - 1)
        self.assertEqual(x_pos_last[0], 70.0)

        # Test invalid index
        with self.assertRaises(ValueError):
            self.data_handler.get_particle_trajectory(100)

    def test_get_all_trajectories(self):
        """Test retrieving all particle trajectories."""
        # Record data with all particles
        self.data_handler.record_state(0.0, 100.0, self.particles)
        self.data_handler.record_state(0.1, 99.0, self.particles)

        trajectories = self.data_handler.get_all_trajectories()

        # Should return trajectories for all particles
        self.assertEqual(len(trajectories), const.N_PARTICLES)

        # Each trajectory should have correct structure
        for i, (x_pos, y_pos) in enumerate(trajectories):
            self.assertEqual(len(x_pos), 2)
            self.assertEqual(len(y_pos), 2)
            self.assertIsInstance(x_pos[0], float)
            self.assertIsInstance(y_pos[0], float)

    def test_get_energy_history(self):
        """Test retrieving energy history."""
        # Record energy evolution
        energies = [100.0, 99.5, 99.2, 99.0]
        times = [0.0, 0.1, 0.2, 0.3]

        for t, e in zip(times, energies):
            self.data_handler.record_state(t, e, self.particles)

        t_history, e_history = self.data_handler.get_energy_history()

        self.assertEqual(t_history, times)
        self.assertEqual(e_history, energies)

    def test_load_from_file(self):
        """Test loading data from CSV file."""
        # Create proper CSV with all particles
        for t in range(3):
            self.data_handler.record_state(t * 0.1, 100.0 - t, self.particles)
        self.data_handler.save()

        # Load with new handler
        new_handler = DataHandler()
        new_handler.load_from_file(self.test_file)

        # Verify loaded data
        self.assertEqual(len(new_handler.trajectory_data['time']), 3)
        self.assertEqual(new_handler.trajectory_data['time'][0], 0.0)
        self.assertEqual(new_handler.trajectory_data['energy'][2], 98.0)

        # Verify all particle data loaded
        self.assertEqual(
            len(new_handler.trajectory_data['particles'][0]), const.N_PARTICLES)

    def test_get_statistics(self):
        """Test statistics calculation with proper numerical precision."""
        # Record data with energy evolution
        energies = [100.0, 99.8, 99.7, 99.5, 99.3]
        for i, e in enumerate(energies):
            self.data_handler.record_state(i * 0.1, e, self.particles)

        stats = self.data_handler.get_statistics()

        # Verify statistics with appropriate precision
        self.assertEqual(stats['initial_energy'], 100.0)
        self.assertEqual(stats['final_energy'], 99.3)

        # Handle floating point precision properly
        self.assertAlmostEqual(stats['energy_drift'], -0.7, places=10)
        self.assertAlmostEqual(stats['relative_drift'], -0.007, places=10)

        self.assertEqual(stats['num_timesteps'], 5)
        self.assertAlmostEqual(
            stats['mean_energy'], np.mean(energies), places=10)
        self.assertAlmostEqual(
            stats['std_energy'], np.std(energies), places=10)

    def test_empty_statistics(self):
        """Test statistics with no data."""
        stats = self.data_handler.get_statistics()
        self.assertEqual(stats, {})

    def test_output_format_specification(self):
        """Test output format matches project specification."""
        # Record one complete timestep
        self.data_handler.record_state(1.234, 567.89, self.particles)

        # Verify buffer format
        row = self.data_handler.data_buffer[0]

        # Check total column count
        expected_length = 2 + 4 * const.N_PARTICLES
        self.assertEqual(len(row), expected_length)

        # Verify time and energy columns
        self.assertEqual(row[0], 1.234)
        self.assertEqual(row[1], 567.89)

        # Verify particle data columns follow specification format
        # Format: t, E_total, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...
        self.assertEqual(row[2], 10.0)   # x1
        self.assertEqual(row[3], 20.0)   # y1
        self.assertEqual(row[4], 5.0)    # vx1
        self.assertEqual(row[5], -3.0)   # vy1
        self.assertEqual(row[6], 20.0)   # x2
        self.assertEqual(row[7], 25.0)   # y2

    def test_file_io_error_handling(self):
        """Test graceful handling of file I/O errors."""
        # Attempt save to invalid path
        invalid_handler = DataHandler("/invalid/path/file.csv")
        invalid_handler.record_state(0.0, 100.0, self.particles)

        # Should handle error gracefully without crashing
        invalid_handler.save()

        # Test loading non-existent file
        invalid_handler.load_from_file("/nonexistent/file.csv")

    def test_large_dataset(self):
        """Test handling of large datasets."""
        # Record many timesteps
        n_steps = 1000
        for i in range(n_steps):
            self.data_handler.record_state(i * 0.001, 100.0, self.particles)

        # Verify data structure
        self.assertEqual(len(self.data_handler.data_buffer), n_steps)
        self.assertEqual(self.data_handler.records_written, n_steps)

        # Save and verify
        self.data_handler.save()
        self.assertTrue(os.path.exists(self.test_file))

    def test_matches_main_simulation_output(self):
        """Test that data handler produces output matching main simulation."""
        # Use actual initial states from main simulation
        particles = []
        for state in const.INITIAL_STATES:
            particle = Particle(
                x=state[0], y=state[1],
                vx=state[2], vy=state[3],
                mass=const.MASS, charge=const.CHARGE
            )
            particles.append(particle)

        # Record state like main simulation does
        time = 1.234
        energy = 6658.979403  # Similar to main simulation initial energy

        self.data_handler.record_state(time, energy, particles)

        # Verify proper recording
        self.assertEqual(len(self.data_handler.data_buffer), 1)
        row = self.data_handler.data_buffer[0]

        # Should match specification exactly
        self.assertEqual(row[0], time)
        self.assertEqual(row[1], energy)

        # Particle data should match initial states
        self.assertEqual(row[2], 1.0)    # x1 from INITIAL_STATES[0][0]
        self.assertEqual(row[3], 45.0)   # y1 from INITIAL_STATES[0][1]
        self.assertEqual(row[4], 10.0)   # vx1 from INITIAL_STATES[0][2]
        self.assertEqual(row[5], 0.0)    # vy1 from INITIAL_STATES[0][3]

    def test_trajectory_data_consistency(self):
        """Test that trajectory data remains consistent across operations."""
        # Record several timesteps
        for t in range(10):
            # Modify particle positions slightly each step
            for i, particle in enumerate(self.particles):
                particle.state[0] = 10.0 + i * 10.0 + t * 0.1  # x
                particle.state[1] = 20.0 + i * 5.0 + t * 0.05  # y

            self.data_handler.record_state(
                t * 0.01, 1000.0 - t, self.particles)

        # Verify all trajectory access methods work consistently
        all_trajectories = self.data_handler.get_all_trajectories()

        for i in range(const.N_PARTICLES):
            individual_trajectory = self.data_handler.get_particle_trajectory(
                i)

            # Individual and bulk access should give same result
            np.testing.assert_array_equal(
                individual_trajectory[0], all_trajectories[i][0])
            np.testing.assert_array_equal(
                individual_trajectory[1], all_trajectories[i][1])

    def test_energy_history_consistency(self):
        """Test energy history data consistency."""
        energies = [1000.0, 999.5, 999.1, 998.8, 998.5]
        times = [0.0, 0.1, 0.2, 0.3, 0.4]

        for t, e in zip(times, energies):
            self.data_handler.record_state(t, e, self.particles)

        # Get energy history
        t_hist, e_hist = self.data_handler.get_energy_history()

        # Should match input exactly
        self.assertEqual(len(t_hist), len(times))
        self.assertEqual(len(e_hist), len(energies))

        for i in range(len(times)):
            self.assertEqual(t_hist[i], times[i])
            self.assertEqual(e_hist[i], energies[i])

    def test_statistics_precision(self):
        """Test statistics calculation with proper floating point handling."""
        # Use simple values that avoid floating point precision issues
        energies = [100.0, 99.0, 98.0, 97.0, 96.0]
        for i, e in enumerate(energies):
            self.data_handler.record_state(i * 0.1, e, self.particles)

        stats = self.data_handler.get_statistics()

        # Verify basic statistics
        self.assertEqual(stats['initial_energy'], 100.0)
        self.assertEqual(stats['final_energy'], 96.0)
        self.assertEqual(stats['energy_drift'], -4.0)
        self.assertEqual(stats['relative_drift'], -0.04)
        self.assertEqual(stats['num_timesteps'], 5)

        # Verify computed statistics
        expected_mean = np.mean(energies)
        expected_std = np.std(energies)
        self.assertAlmostEqual(stats['mean_energy'], expected_mean, places=10)
        self.assertAlmostEqual(stats['std_energy'], expected_std, places=10)

    def test_csv_round_trip(self):
        """Test complete save/load cycle preserves data."""
        # Record original data
        original_times = []
        original_energies = []

        for t in range(5):
            time_val = t * 0.1
            energy_val = 1000.0 - t * 0.5

            self.data_handler.record_state(
                time_val, energy_val, self.particles)
            original_times.append(time_val)
            original_energies.append(energy_val)

        # Save to file
        self.data_handler.save()

        # Load with new handler
        new_handler = DataHandler()
        new_handler.load_from_file(self.test_file)

        # Compare original and loaded data
        loaded_times, loaded_energies = new_handler.get_energy_history()

        np.testing.assert_array_equal(loaded_times, original_times)
        np.testing.assert_array_equal(loaded_energies, original_energies)

        # Verify particle data preservation
        for i in range(const.N_PARTICLES):
            orig_traj = self.data_handler.get_particle_trajectory(i)
            loaded_traj = new_handler.get_particle_trajectory(i)

            np.testing.assert_array_equal(orig_traj[0], loaded_traj[0])
            np.testing.assert_array_equal(orig_traj[1], loaded_traj[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
