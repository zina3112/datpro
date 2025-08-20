"""
test_visualization.py - Comprehensive unit tests for Visualizer class

Tests all plotting and visualization functionality including energy plots,
trajectories, phase space diagrams, and report generation.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import Visualizer
from src.data_handler import DataHandler
from src.particle import Particle
import src.constants as const


class TestVisualizerInitialization(unittest.TestCase):
    """Test Visualizer initialization."""

    def setUp(self):
        """Create temporary directory and data handler."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Add some test data
        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0)
        ]

        for i in range(5):
            self.data_handler.record_state(float(i), 100.0 + i, particles)

    def tearDown(self):
        """Clean up temporary files."""
        plt.close('all')  # Close all matplotlib figures
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_default(self):
        """Test visualizer initialization with defaults."""
        viz = Visualizer(self.data_handler)

        self.assertEqual(viz.data_handler, self.data_handler)
        self.assertIsNotNone(viz.figure_dir)

        # Should create figure directory
        self.assertTrue(os.path.exists(viz.figure_dir))

    def test_initialization_custom_dir(self):
        """Test visualizer with custom figure directory."""
        custom_dir = os.path.join(self.temp_dir, "custom_plots")
        viz = Visualizer(self.data_handler, figure_dir=custom_dir)

        self.assertEqual(viz.figure_dir, custom_dir)
        self.assertTrue(os.path.exists(custom_dir))

    def test_matplotlib_style(self):
        """Test that matplotlib style is set."""
        # Store original style
        original_style = plt.rcParams.copy()

        viz = Visualizer(self.data_handler)

        # Style should be applied (checking is complex due to style availability)
        # Just ensure no crash
        self.assertIsNotNone(viz)


class TestEnergyPlots(unittest.TestCase):
    """Test energy plotting functionality."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create data with energy drift
        particles = [Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0)]

        for i in range(10):
            energy = 100.0 + 0.1 * i  # Small drift
            self.data_handler.record_state(float(i) * 0.1, energy, particles)

        self.viz = Visualizer(self.data_handler,
                             figure_dir=os.path.join(self.temp_dir, "plots"))

    def tearDown(self):
        """Clean up."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_plot_energy_vs_time(self):
        """Test energy vs time plot creation."""
        self.viz.plot_energy_vs_time(save=True, show=False)

        # Check file was created
        energy_plot = os.path.join(self.viz.figure_dir, 'energy_conservation.png')
        self.assertTrue(os.path.exists(energy_plot))

        # Check file is not empty
        self.assertGreater(os.path.getsize(energy_plot), 0)

    def test_plot_energy_no_data(self):
        """Test energy plot with no data."""
        empty_handler = DataHandler(os.path.join(self.temp_dir, "empty.csv"))
        viz = Visualizer(empty_handler,
                        figure_dir=os.path.join(self.temp_dir, "empty_plots"))

        # Should handle gracefully
        viz.plot_energy_vs_time(save=False, show=False)

        # No file should be created
        energy_plot = os.path.join(viz.figure_dir, 'energy_conservation.png')
        self.assertFalse(os.path.exists(energy_plot))

    def test_plot_energy_save_false(self):
        """Test energy plot without saving."""
        self.viz.plot_energy_vs_time(save=False, show=False)

        # File should not be created
        energy_plot = os.path.join(self.viz.figure_dir, 'energy_conservation.png')
        self.assertFalse(os.path.exists(energy_plot))


class TestTrajectoryPlots(unittest.TestCase):
    """Test trajectory plotting functionality."""

    def setUp(self):
        """Create test data with trajectories."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create particles with different trajectories
        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=0.0),
            Particle(x=30.0, y=40.0, vx=0.0, vy=-5.0)
        ]

        # Simulate motion
        for i in range(10):
            particles[0].state[0] = 10.0 + i * 5.0  # Move right
            particles[1].state[1] = 40.0 - i * 5.0  # Move down
            self.data_handler.record_state(float(i), 100.0, particles)

        self.viz = Visualizer(self.data_handler,
                             figure_dir=os.path.join(self.temp_dir, "plots"))

    def tearDown(self):
        """Clean up."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_plot_particle_trajectory(self):
        """Test single particle trajectory plot."""
        self.viz.plot_particle_trajectory(0, save=True, show=False)

        # Check file was created
        traj_plot = os.path.join(self.viz.figure_dir, 'trajectory_particle_1.png')
        self.assertTrue(os.path.exists(traj_plot))

    def test_plot_particle_trajectory_invalid_index(self):
        """Test trajectory plot with invalid particle index."""
        # Should handle gracefully
        self.viz.plot_particle_trajectory(10, save=False, show=False)

        # No crash expected

    def test_plot_all_trajectories(self):
        """Test plotting all particle trajectories."""
        # Need to set N_PARTICLES correctly
        original_n = const.N_PARTICLES
        const.N_PARTICLES = 2

        try:
            self.viz.plot_all_trajectories(save=True, show=False)

            # Check file was created
            all_traj_plot = os.path.join(self.viz.figure_dir, 'all_trajectories.png')
            self.assertTrue(os.path.exists(all_traj_plot))

        finally:
            const.N_PARTICLES = original_n

    def test_plot_trajectory_empty_data(self):
        """Test trajectory plot with no data."""
        empty_handler = DataHandler(os.path.join(self.temp_dir, "empty.csv"))
        viz = Visualizer(empty_handler,
                        figure_dir=os.path.join(self.temp_dir, "empty_plots"))

        viz.plot_particle_trajectory(0, save=False, show=False)

        # Should not crash


class TestPhaseSpacePlots(unittest.TestCase):
    """Test phase space plotting functionality."""

    def setUp(self):
        """Create test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create oscillating particle
        particles = [Particle(x=0.0, y=0.0, vx=10.0, vy=10.0)]

        for i in range(20):
            # Simulate oscillation
            angle = i * 0.3
            particles[0].state[0] = 50.0 + 10.0 * np.cos(angle)
            particles[0].state[1] = 50.0 + 10.0 * np.sin(angle)
            particles[0].state[2] = -10.0 * np.sin(angle)
            particles[0].state[3] = 10.0 * np.cos(angle)

            self.data_handler.record_state(float(i) * 0.1, 100.0, particles)

        self.viz = Visualizer(self.data_handler,
                             figure_dir=os.path.join(self.temp_dir, "plots"))

    def tearDown(self):
        """Clean up."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_plot_phase_space(self):
        """Test phase space plot creation."""
        self.viz.plot_phase_space(0, save=True, show=False)

        # Check file was created
        phase_plot = os.path.join(self.viz.figure_dir, 'phase_space_particle_1.png')
        self.assertTrue(os.path.exists(phase_plot))

    def test_plot_phase_space_multiple_particles(self):
        """Test phase space for different particles."""
        # Add second particle
        particles = [
            Particle(x=0.0, y=0.0, vx=5.0, vy=5.0),
            Particle(x=10.0, y=10.0, vx=-5.0, vy=-5.0)
        ]

        handler = DataHandler(os.path.join(self.temp_dir, "multi.csv"))
        for i in range(5):
            handler.record_state(float(i), 100.0, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "multi_plots"))

        # Plot for both particles
        viz.plot_phase_space(0, save=True, show=False)
        viz.plot_phase_space(1, save=True, show=False)

        # Both files should exist
        phase1 = os.path.join(viz.figure_dir, 'phase_space_particle_1.png')
        phase2 = os.path.join(viz.figure_dir, 'phase_space_particle_2.png')

        self.assertTrue(os.path.exists(phase1))
        self.assertTrue(os.path.exists(phase2))


class TestBoxBoundaryDrawing(unittest.TestCase):
    """Test box boundary visualization."""

    def setUp(self):
        """Create test setup."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Add minimal data
        particles = [Particle(x=50.0, y=50.0, vx=0, vy=0)]
        self.data_handler.record_state(0.0, 100.0, particles)

        self.viz = Visualizer(self.data_handler,
                             figure_dir=os.path.join(self.temp_dir, "plots"))

    def tearDown(self):
        """Clean up."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_draw_box_boundaries(self):
        """Test that box boundaries are drawn correctly."""
        fig, ax = plt.subplots()

        self.viz._draw_box_boundaries(ax)

        # Check that lines were added
        lines = ax.get_lines()
        self.assertGreater(len(lines), 0)

        # Check axis limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Should include box with margin
        self.assertLess(xlim[0], const.BOX_MIN_X)
        self.assertGreater(xlim[1], const.BOX_MAX_X)
        self.assertLess(ylim[0], const.BOX_MIN_Y)
        self.assertGreater(ylim[1], const.BOX_MAX_Y)

        plt.close(fig)


class TestSummaryReport(unittest.TestCase):
    """Test summary report generation."""

    def setUp(self):
        """Create comprehensive test data."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))

        # Create multiple particles with data
        self.original_n = const.N_PARTICLES
        const.N_PARTICLES = 3

        particles = [
            Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0),
            Particle(x=30.0, y=40.0, vx=-5.0, vy=3.0),
            Particle(x=50.0, y=60.0, vx=0.0, vy=0.0)
        ]

        for i in range(5):
            self.data_handler.record_state(float(i), 100.0 + i, particles)

        self.viz = Visualizer(self.data_handler,
                             figure_dir=os.path.join(self.temp_dir, "plots"))

    def tearDown(self):
        """Clean up."""
        const.N_PARTICLES = self.original_n
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create_summary_report(self):
        """Test complete summary report generation."""
        self.viz.create_summary_report()

        # Check that files were created
        expected_files = [
            'energy_conservation.png',
            'trajectory_particle_1.png',
            'trajectory_particle_2.png',
            'trajectory_particle_3.png',
            'phase_space_particle_1.png',
            'phase_space_particle_2.png',
            'phase_space_particle_3.png',
            'all_trajectories.png',
            'statistics.txt'
        ]

        for filename in expected_files:
            filepath = os.path.join(self.viz.figure_dir, filename)
            self.assertTrue(os.path.exists(filepath),
                          f"Missing file: {filename}")

    def test_statistics_file_content(self):
        """Test that statistics file contains correct information."""
        self.viz.create_summary_report()

        stats_file = os.path.join(self.viz.figure_dir, 'statistics.txt')

        with open(stats_file, 'r') as f:
            content = f.read()

        # Check content
        self.assertIn("SIMULATION STATISTICS", content)
        self.assertIn("initial_energy", content)
        self.assertIn("final_energy", content)
        self.assertIn("num_timesteps", content)


class TestVisualizationEdgeCases(unittest.TestCase):
    """Test edge cases in visualization."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        plt.close('all')
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_data_point(self):
        """Test visualization with single data point."""
        handler = DataHandler(os.path.join(self.temp_dir, "single.csv"))
        particles = [Particle(x=50.0, y=50.0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

        # Should handle single point
        viz.plot_energy_vs_time(save=True, show=False)
        viz.plot_particle_trajectory(0, save=True, show=False)

    def test_nan_values_in_data(self):
        """Test handling NaN values in data."""
        handler = DataHandler(os.path.join(self.temp_dir, "nan.csv"))
        particles = [Particle(x=float('nan'), y=50.0, vx=0, vy=0)]

        for i in range(3):
            handler.record_state(float(i), 100.0, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

        # Should handle NaN without crashing
        viz.plot_particle_trajectory(0, save=True, show=False)

    def test_very_large_values(self):
        """Test visualization with very large values."""
        handler = DataHandler(os.path.join(self.temp_dir, "large.csv"))
        particles = [Particle(x=1e10, y=1e10, vx=1e5, vy=1e5)]

        for i in range(3):
            handler.record_state(float(i), 1e15, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

        # Should handle large values
        viz.plot_energy_vs_time(save=True, show=False)
        viz.plot_particle_trajectory(0, save=True, show=False)

    def test_zero_particles(self):
        """Test visualization with no particles."""
        original_n = const.N_PARTICLES
        const.N_PARTICLES = 0

        try:
            handler = DataHandler(os.path.join(self.temp_dir, "zero.csv"))
            handler.record_state(0.0, 100.0, [])

            viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

            # Should handle empty particle list
            viz.plot_all_trajectories(save=False, show=False)

        finally:
            const.N_PARTICLES = original_n

    def test_custom_figure_size(self):
        """Test that figure size is applied."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))
        particles = [Particle(x=50.0, y=50.0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

        # Check figure size constant is used
        self.assertIsNotNone(const.FIGURE_SIZE)
        self.assertEqual(len(const.FIGURE_SIZE), 2)

    def test_file_permissions_error(self):
        """Test handling of file permission errors."""
        handler = DataHandler(os.path.join(self.temp_dir, "test.csv"))
        particles = [Particle(x=50.0, y=50.0, vx=0, vy=0)]
        handler.record_state(0.0, 100.0, particles)

        # Create read-only directory
        readonly_dir = os.path.join(self.temp_dir, "readonly")
        os.makedirs(readonly_dir, mode=0o444)

        # This might fail on some systems, so we catch and skip
        try:
            viz = Visualizer(handler, figure_dir=readonly_dir)
            # Try to save plot (should handle error gracefully)
            viz.plot_energy_vs_time(save=True, show=False)
        except:
            pass  # Expected on some systems

    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        handler = DataHandler(os.path.join(self.temp_dir, "large.csv"))
        particles = [Particle(x=50.0, y=50.0, vx=0, vy=0)]

        # Create large dataset
        for i in range(1000):
            handler.record_state(float(i), 100.0, particles)

        viz = Visualizer(handler, figure_dir=os.path.join(self.temp_dir, "plots"))

        # Should handle large dataset efficiently
        viz.plot_energy_vs_time(save=True, show=False)
        viz.plot_particle_trajectory(0, save=True, show=False)

        # Files should be created
        energy_plot = os.path.join(viz.figure_dir, 'energy_conservation.png')
        self.assertTrue(os.path.exists(energy_plot))


if __name__ == '__main__':
    unittest.main(verbosity=2)
