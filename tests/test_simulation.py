"""
test_simulation.py - Comprehensive unit tests for main simulation controller

Tests the complete simulation workflow including initialization,
time stepping, collision handling, and data recording.
"""

import src.constants as const
from src.particle import Particle
from src.simulation import Simulation
import unittest
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSimulationInitialization(unittest.TestCase):
    """Test simulation initialization."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_initialization_default(self):
        """Test simulation with default initial conditions."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(output_file=output_file)

        # Should use default initial states from constants
        self.assertEqual(len(sim.particles), len(const.INITIAL_STATES))
        self.assertEqual(sim.dt, const.DT)
        self.assertEqual(sim.current_time, 0.0)
        self.assertEqual(sim.step_count, 0)

    def test_initialization_custom_states(self):
        """Test simulation with custom initial states."""
        custom_states = np.array([
            [10.0, 20.0, 5.0, -3.0],
            [30.0, 40.0, -5.0, 3.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=custom_states, output_file=output_file)

        self.assertEqual(len(sim.particles), 2)
        self.assertEqual(sim.particles[0].x, 10.0)
        self.assertEqual(sim.particles[0].y, 20.0)
        self.assertEqual(sim.particles[1].x, 30.0)
        self.assertEqual(sim.particles[1].y, 40.0)

    def test_initialization_custom_timestep(self):
        """Test simulation with custom timestep."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(dt=0.002, output_file=output_file)

        self.assertEqual(sim.dt, 0.002)
        self.assertEqual(sim.integrator.dt, 0.002)

    def test_initialization_particles_properties(self):
        """Test that particles are initialized with correct properties."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(output_file=output_file)

        for particle in sim.particles:
            self.assertEqual(particle.mass, const.MASS)
            self.assertEqual(particle.charge, const.CHARGE)

    def test_initialization_energy_calculation(self):
        """Test initial energy is calculated correctly."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(output_file=output_file)

        self.assertIsNotNone(sim.initial_energy)
        self.assertGreater(sim.initial_energy, 0)
        self.assertEqual(len(sim.energy_history), 1)
        self.assertEqual(sim.energy_history[0], sim.initial_energy)

    def test_initialization_data_recording(self):
        """Test that initial state is recorded."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(output_file=output_file)

        # Initial state should be recorded
        self.assertEqual(len(sim.data_handler.data_buffer), 1)
        self.assertEqual(sim.data_handler.data_buffer[0][0], 0.0)  # Time = 0


class TestSimulationStepping(unittest.TestCase):
    """Test simulation time stepping."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_step(self):
        """Test single simulation step."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        initial_x = sim.particles[0].x
        initial_y = sim.particles[0].y

        success = sim.step()

        self.assertTrue(success)
        self.assertEqual(sim.step_count, 1)
        self.assertAlmostEqual(sim.current_time, 0.001)

        # Particle should have moved
        self.assertGreater(sim.particles[0].x, initial_x)
        self.assertLess(sim.particles[0].y, initial_y)  # Gravity

    def test_multiple_steps(self):
        """Test multiple simulation steps."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        for _ in range(10):
            success = sim.step()
            self.assertTrue(success)

        self.assertEqual(sim.step_count, 10)
        self.assertAlmostEqual(sim.current_time, 0.01)

        # Should have 11 data points (initial + 10 steps)
        self.assertEqual(len(sim.data_handler.data_buffer), 11)

    def test_step_with_collision(self):
        """Test stepping with wall collision."""
        # Particle near wall moving toward it
        initial_states = np.array([
            [99.5, 50.0, 100.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.01, output_file=output_file)

        # Step until collision
        for _ in range(5):
            sim.step()

        # Should have collision
        self.assertGreater(sim.particles[0].collision_count, 0)

        # Should stay inside box
        self.assertLessEqual(sim.particles[0].x, const.BOX_MAX_X)
        self.assertGreaterEqual(sim.particles[0].x, const.BOX_MIN_X)

    def test_step_energy_tracking(self):
        """Test that energy is tracked each step."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        initial_energy_count = len(sim.energy_history)

        sim.step()

        self.assertEqual(len(sim.energy_history), initial_energy_count + 1)
        self.assertEqual(len(sim.time_history), initial_energy_count + 1)


class TestSimulationRun(unittest.TestCase):
    """Test complete simulation runs."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_run_basic(self):
        """Test basic simulation run."""
        initial_states = np.array([
            [50.0, 50.0, 5.0, 0.0],
            [30.0, 30.0, -5.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.01, output_file=output_file)

        sim.run(simulation_time=0.1, progress_interval=10)

        self.assertAlmostEqual(sim.current_time, 0.1, places=5)
        self.assertEqual(sim.step_count, 10)

        # Data should be saved
        self.assertTrue(os.path.exists(output_file))

    def test_run_with_progress(self):
        """Test simulation run with progress updates."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Capture progress output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        sim.run(simulation_time=0.01, progress_interval=5)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Should have progress updates
        self.assertIn("Progress:", output)
        self.assertIn("100", output)  # 100% completion

    def test_run_standard_configuration(self):
        """Test run with standard 7-particle configuration."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(dt=0.001, output_file=output_file)

        # Short run
        sim.run(simulation_time=0.01, progress_interval=100)

        self.assertEqual(len(sim.particles), 7)
        self.assertAlmostEqual(sim.current_time, 0.01)

        # All particles should have moved
        for i, particle in enumerate(sim.particles):
            initial_state = const.INITIAL_STATES[i]
            self.assertNotEqual(particle.x, initial_state[0])

    def test_run_timing_statistics(self):
        """Test that timing statistics are calculated."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        sim.run(simulation_time=0.01, progress_interval=100)

        self.assertIsNotNone(sim.start_real_time)
        self.assertGreater(sim.total_computation_time, 0)


class TestEnergyCalculation(unittest.TestCase):
    """Test energy calculation methods."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_total_energy_single_particle(self):
        """Test energy calculation for single particle."""
        initial_states = np.array([
            [0.0, 10.0, 3.0, 4.0]  # KE = 12.5, PE_grav = 100
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        energy = sim.calculate_total_energy()

        # KE = 0.5 * 1 * (9 + 16) = 12.5
        # PE_grav = -1 * (-10) * 10 = 100
        # No Coulomb energy for single particle
        expected = 112.5
        self.assertAlmostEqual(energy, expected, places=5)

    def test_total_energy_two_particles(self):
        """Test energy calculation for two particles."""
        initial_states = np.array([
            [0.0, 50.0, 10.0, 0.0],
            [10.0, 50.0, -10.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Set known charges for calculation
        sim.particles[0].charge = 10.0
        sim.particles[1].charge = 10.0

        energy = sim.calculate_total_energy()

        # KE = 0.5 * 1 * 100 * 2 = 100
        # PE_grav = -1 * (-10) * 50 * 2 = 1000
        # PE_coulomb = 10 * 10 / 10 = 10
        expected = 100 + 1000 + 10
        self.assertAlmostEqual(energy, expected, places=5)

    def test_energy_components(self):
        """Test that energy components are calculated correctly."""
        initial_states = np.array([
            [0.0, 100.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0  # No Coulomb

        energy = sim.calculate_total_energy()

        # Only gravitational PE
        expected = -const.MASS * const.GRAVITY * 100.0
        self.assertAlmostEqual(energy, expected, places=5)


class TestTrajectoryRetrieval(unittest.TestCase):
    """Test trajectory and data retrieval methods."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_get_trajectory(self):
        """Test getting particle trajectory."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Run for a few steps
        for _ in range(10):
            sim.step()

        x_pos, y_pos = sim.get_trajectory(0)

        self.assertEqual(len(x_pos), 11)  # Initial + 10 steps
        self.assertEqual(len(y_pos), 11)

        # Should show motion
        self.assertGreater(x_pos[-1], x_pos[0])
        self.assertLess(y_pos[-1], y_pos[0])  # Gravity

    def test_get_energy_history(self):
        """Test getting energy history."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Run for a few steps
        for _ in range(5):
            sim.step()

        times, energies = sim.get_energy_history()

        self.assertEqual(len(times), 6)  # Initial + 5 steps
        self.assertEqual(len(energies), 6)
        self.assertEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 0.005)


class TestStatistics(unittest.TestCase):
    """Test simulation statistics generation."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_print_statistics(self):
        """Test statistics printing."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 0.0],
            [30.0, 30.0, -10.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Run simulation
        sim.run(simulation_time=0.01, progress_interval=100)

        # Capture statistics output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        sim.print_statistics()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Check for expected statistics
        self.assertIn("SIMULATION STATISTICS", output)
        self.assertIn("Simulation time:", output)
        self.assertIn("Energy Conservation:", output)
        self.assertIn("Initial energy:", output)
        self.assertIn("Final energy:", output)
        self.assertIn("Collision Statistics:", output)

    def test_collision_statistics(self):
        """Test collision counting."""
        # Particle that will hit wall
        initial_states = np.array([
            [99.0, 50.0, 50.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Run until collision
        for _ in range(50):
            sim.step()

        # Should have collision
        total_collisions = sum(p.collision_count for p in sim.particles)
        self.assertGreater(total_collisions, 0)


class TestSimulationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Create temporary directory for output files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_particle_simulation(self):
        """Test simulation with single particle."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        sim.run(simulation_time=0.01, progress_interval=100)

        # Should complete successfully
        self.assertAlmostEqual(sim.current_time, 0.01)

    def test_many_particles_simulation(self):
        """Test simulation with many particles."""
        # Create 20 particles
        n = 20
        initial_states = np.array([
            [float(i * 5), float(i * 5), 0.0, 0.0] for i in range(n)
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        self.assertEqual(len(sim.particles), n)

        # Should be able to step
        success = sim.step()
        self.assertTrue(success)

    def test_zero_timestep(self):
        """Test simulation with zero timestep."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0, output_file=output_file)

        initial_pos = sim.particles[0].position.copy()

        sim.step()

        # Should not move
        np.testing.assert_array_equal(sim.particles[0].position, initial_pos)

    def test_very_small_timestep(self):
        """Test simulation with very small timestep."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=1e-10, output_file=output_file)

        sim.step()

        # Should have very small changes
        self.assertAlmostEqual(sim.particles[0].x, 50.0, places=8)

    def test_overlapping_initial_particles(self):
        """Test simulation with initially overlapping particles."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0],
            [50.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Should handle without error
        sim.step()

        # Particles should separate (if they have charge)
        if sim.particles[0].charge > 0:
            dist = sim.particles[0].distance_to(sim.particles[1])
            self.assertGreater(dist, 0)

    def test_data_handler_integration(self):
        """Test data handler is properly integrated."""
        initial_states = np.array([
            [50.0, 50.0, 5.0, 5.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        # Run simulation
        sim.run(simulation_time=0.005, progress_interval=10)

        # Check data was recorded
        stats = sim.data_handler.get_statistics()
        self.assertIn('num_timesteps', stats)
        self.assertEqual(stats['num_timesteps'], 6)  # Initial + 5 steps


if __name__ == '__main__':
    unittest.main(verbosity=2)
