"""
test_simulation.py - Unit tests for main simulation controller

Tests the complete simulation workflow and energy conservation.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import Simulation
from src.particle import Particle
import src.constants as const


class TestSimulation(unittest.TestCase):
    """Test suite for Simulation class."""

    def setUp(self):
        """Set up test simulation."""
        # Create temporary directory for output files
        self.test_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.test_dir, "test_output.csv")

        # Simple test configuration
        self.test_states = np.array([
            [10.0, 50.0, 5.0, 0.0],
            [90.0, 50.0, -5.0, 0.0]
        ])

        self.sim = Simulation(
            initial_states=self.test_states,
            dt=0.001,
            output_file=self.output_file
        )

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test simulation initialization."""
        # Check particles created correctly
        self.assertEqual(len(self.sim.particles), 2)

        # Check initial positions
        self.assertEqual(self.sim.particles[0].x, 10.0)
        self.assertEqual(self.sim.particles[1].x, 90.0)

        # Check timestep
        self.assertEqual(self.sim.dt, 0.001)

        # Check initial energy calculated
        self.assertIsNotNone(self.sim.initial_energy)
        self.assertTrue(np.isfinite(self.sim.initial_energy))

    def test_energy_calculation(self):
        """Test total energy calculation."""
        energy = self.sim.calculate_total_energy()

        # Energy should be finite
        self.assertTrue(np.isfinite(energy))

        # Energy should have contributions from:
        # - Kinetic energy (particles are moving)
        # - Gravitational potential (particles have height)
        # - Coulomb potential (particles repel)

        # Rough check: energy shouldn't be zero
        self.assertNotEqual(energy, 0.0)

    def test_single_step(self):
        """Test single simulation step."""
        initial_energy = self.sim.calculate_total_energy()
        initial_positions = [p.position.copy() for p in self.sim.particles]

        # Perform one step
        success = self.sim.step()
        self.assertTrue(success)

        # Check time advanced
        self.assertAlmostEqual(self.sim.current_time, self.sim.dt)
        self.assertEqual(self.sim.step_count, 1)

        # Check particles moved
        for i, p in enumerate(self.sim.particles):
            # Particles should have moved (unless at equilibrium, which they're not)
            pos_change = np.linalg.norm(p.position - initial_positions[i])
            self.assertGreater(pos_change, 0.0)

        # Check energy conservation (should be within tolerance)
        final_energy = self.sim.calculate_total_energy()
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(energy_drift, 0.01)  # Less than 1% drift per step

    def test_multiple_steps(self):
        """Test multiple simulation steps."""
        n_steps = 10

        for _ in range(n_steps):
            success = self.sim.step()
            self.assertTrue(success)

        self.assertEqual(self.sim.step_count, n_steps)
        self.assertAlmostEqual(self.sim.current_time, n_steps * self.sim.dt)

    def test_energy_conservation_short_run(self):
        """Test energy conservation over short simulation."""
        # Run for 100 steps
        for _ in range(100):
            self.sim.step()

        final_energy = self.sim.calculate_total_energy()
        energy_drift = abs(final_energy - self.sim.initial_energy)
        relative_drift = energy_drift / abs(self.sim.initial_energy)

        # For 100 steps with dt=0.001, drift should be very small
        self.assertLess(relative_drift, 0.001)  # Less than 0.1% total drift

    def test_wall_collisions_occur(self):
        """Test that wall collisions are detected and handled."""
        # Create particle that will definitely hit wall
        states = np.array([[95.0, 50.0, 20.0, 0.0]])  # Fast rightward motion

        sim = Simulation(initial_states=states, dt=0.1, output_file=self.output_file)

        # Run until collision should occur
        for _ in range(10):
            sim.step()

        # Check collision was recorded
        self.assertGreater(sim.box.total_collisions, 0)

    def test_particle_stays_in_box(self):
        """Test that particles always stay within box boundaries."""
        # Run simulation with various initial conditions
        states = np.array([
            [95.0, 95.0, 30.0, 30.0],  # Corner-bound
            [5.0, 5.0, -30.0, -30.0],   # Other corner
            [50.0, 99.0, 0.0, 20.0]     # Top wall
        ])

        sim = Simulation(initial_states=states, dt=0.01, output_file=self.output_file)

        # Run for many steps
        for _ in range(100):
            sim.step()

            # Check all particles remain in box
            for particle in sim.particles:
                self.assertGreaterEqual(particle.x, const.BOX_MIN_X)
                self.assertLessEqual(particle.x, const.BOX_MAX_X)
                self.assertGreaterEqual(particle.y, const.BOX_MIN_Y)
                self.assertLessEqual(particle.y, const.BOX_MAX_Y)

    def test_data_recording(self):
        """Test that simulation data is recorded correctly."""
        # Run a few steps
        n_steps = 5
        for _ in range(n_steps):
            self.sim.step()

        # Check data was recorded
        self.assertEqual(len(self.sim.data_handler.trajectory_data['time']), n_steps + 1)  # +1 for initial
        self.assertEqual(len(self.sim.data_handler.trajectory_data['energy']), n_steps + 1)

        # Check times are correct
        expected_times = [i * self.sim.dt for i in range(n_steps + 1)]
        actual_times = self.sim.data_handler.trajectory_data['time']
        for exp, act in zip(expected_times, actual_times):
            self.assertAlmostEqual(exp, act)

    def test_actual_initial_conditions(self):
        """Test with the actual project initial conditions."""
        sim = Simulation(
            initial_states=const.INITIAL_STATES,
            dt=const.DT,
            output_file=self.output_file
        )

        # Check all 7 particles created
        self.assertEqual(len(sim.particles), 7)

        # Run for a short time
        for _ in range(100):
            sim.step()

        # System should remain stable
        final_energy = sim.calculate_total_energy()
        self.assertTrue(np.isfinite(final_energy))

        # All particles should be in box
        for particle in sim.particles:
            self.assertTrue(sim.box.is_inside(particle.position))

    def test_coulomb_repulsion_effect(self):
        """Test that Coulomb repulsion actually affects trajectories."""
        # Two particles with charge
        states_charged = np.array([
            [45.0, 50.0, 0.0, 0.0],
            [55.0, 50.0, 0.0, 0.0]
        ])

        sim_charged = Simulation(
            initial_states=states_charged,
            dt=0.001,
            output_file=self.output_file
        )

        # Run simulation
        for _ in range(100):
            sim_charged.step()

        # Particles should have moved apart due to repulsion
        final_distance = sim_charged.particles[0].distance_to(sim_charged.particles[1])
        initial_distance = 10.0

        self.assertGreater(final_distance, initial_distance)

    def test_gravity_effect(self):
        """Test that gravity affects particle motion."""
        # Single particle with no charge, only gravity
        states = np.array([[50.0, 80.0, 0.0, 0.0]])  # High position, no velocity

        # Temporarily set charge to 0 to isolate gravity
        original_charge = const.CHARGE
        const.CHARGE = 0.0

        try:
            sim = Simulation(
                initial_states=states,
                dt=0.01,
                output_file=self.output_file
            )

            initial_y = sim.particles[0].y
            initial_vy = sim.particles[0].vy

            # Run for a bit
            for _ in range(10):
                sim.step()

            # Particle should have fallen (y decreased, vy negative)
            self.assertLess(sim.particles[0].y, initial_y)
            self.assertLess(sim.particles[0].vy, initial_vy)

        finally:
            const.CHARGE = original_charge

    def test_statistics_calculation(self):
        """Test simulation statistics."""
        # Run simulation
        for _ in range(50):
            self.sim.step()

        stats = self.sim.data_handler.get_statistics()

        # Check statistics exist and make sense
        self.assertIn('initial_energy', stats)
        self.assertIn('final_energy', stats)
        self.assertIn('energy_drift', stats)
        self.assertIn('relative_drift', stats)

        # Energy drift should be small
        self.assertLess(abs(stats['relative_drift']), 0.01)

    def test_zero_mass_particle(self):
        """Test handling of zero mass particle."""
        # Create particle with zero mass
        states = np.array([[50.0, 50.0, 10.0, 10.0]])

        # Temporarily set mass to 0
        original_mass = const.MASS
        const.MASS = 0.0

        try:
            sim = Simulation(
                initial_states=states,
                dt=0.001,
                output_file=self.output_file
            )

            # Should handle zero mass without crashing
            for _ in range(10):
                success = sim.step()
                self.assertTrue(success)

        finally:
            const.MASS = original_mass

    def test_very_small_timestep(self):
        """Test simulation with very small timestep."""
        sim = Simulation(
            initial_states=self.test_states,
            dt=1e-6,
            output_file=self.output_file
        )

        # Should work with tiny timestep
        for _ in range(10):
            success = sim.step()
            self.assertTrue(success)

        # Energy conservation should be excellent
        final_energy = sim.calculate_total_energy()
        drift = abs(final_energy - sim.initial_energy) / abs(sim.initial_energy)
        self.assertLess(drift, 1e-8)

    def test_alternative_step_method(self):
        """Test the alternative batch step method."""
        # Create two identical simulations
        sim1 = Simulation(
            initial_states=self.test_states,
            dt=0.001,
            output_file=self.output_file
        )

        sim2 = Simulation(
            initial_states=self.test_states,
            dt=0.001,
            output_file=self.output_file + "2"
        )

        # Run one with normal step, other with alternative
        for _ in range(10):
            sim1.step()
            sim2.step_alternative_batch()

        # Results should be similar (not identical due to different order)
        for p1, p2 in zip(sim1.particles, sim2.particles):
            pos_diff = np.linalg.norm(p1.position - p2.position)
            self.assertLess(pos_diff, 0.1)  # Should be close


if __name__ == '__main__':
    unittest.main(verbosity=2)
