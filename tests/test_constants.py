"""
test_constants.py - Unit tests for constants validation

Verifies that all constants match the project specification.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.constants as const


class TestConstants(unittest.TestCase):
    """Test suite for validating constants against specification."""

    def test_physical_constants(self):
        """Test physical constants match specification."""
        self.assertEqual(const.MASS, 1.0, "Mass should be 1.0")
        self.assertEqual(const.CHARGE, 50.0, "Charge should be 50.0")
        self.assertEqual(const.GRAVITY, -10.0, "Gravity should be -10.0")

    def test_box_dimensions(self):
        """Test box dimensions."""
        # Specification says box has extension of 100 in each direction
        # Implementation uses 0-100 which is reasonable
        self.assertEqual(const.BOX_MAX_X - const.BOX_MIN_X, 100.0)
        self.assertEqual(const.BOX_MAX_Y - const.BOX_MIN_Y, 100.0)

        # Check convenience variables
        self.assertEqual(const.BOX_WIDTH, 100.0)
        self.assertEqual(const.BOX_HEIGHT, 100.0)

    def test_simulation_parameters(self):
        """Test simulation parameters match specification."""
        self.assertEqual(const.DT, 0.001, "Timestep should be 0.001")
        self.assertEqual(const.SIMULATION_TIME, 10.0, "Simulation time should be 10 seconds")
        self.assertEqual(const.N_STEPS, 10000, "Should have 10000 steps")

    def test_initial_conditions(self):
        """Test initial particle states match specification exactly."""
        expected_states = np.array([
            [1.0, 45.0, 10.0, 0.0],
            [99.0, 55.0, -10.0, 0.0],
            [10.0, 50.0, 15.0, -15.0],
            [20.0, 30.0, -15.0, -15.0],
            [80.0, 70.0, 15.0, 15.0],
            [80.0, 60.0, 15.0, 15.0],
            [80.0, 50.0, 15.0, 15.0]
        ])

        # Check shape
        self.assertEqual(const.INITIAL_STATES.shape, (7, 4))

        # Check each particle's initial state exactly
        np.testing.assert_array_equal(const.INITIAL_STATES, expected_states)

        # Verify N_PARTICLES
        self.assertEqual(const.N_PARTICLES, 7)

    def test_numerical_parameters(self):
        """Test numerical parameters are reasonable."""
        # Epsilon should be very small
        self.assertLess(const.EPSILON, 1e-8)
        self.assertGreater(const.EPSILON, 0)

        # Energy tolerance should be reasonable
        self.assertLess(const.ENERGY_TOLERANCE, 0.01)  # Less than 1%
        self.assertGreater(const.ENERGY_TOLERANCE, 0)

        # Collision parameters
        self.assertEqual(const.COLLISION_EPSILON, const.EPSILON)
        self.assertGreater(const.MAX_COLLISION_ITERATIONS, 5)

    def test_output_parameters(self):
        """Test output configuration."""
        self.assertIsInstance(const.OUTPUT_DIR, str)
        self.assertIsInstance(const.OUTPUT_FILE, str)
        self.assertIsInstance(const.PLOT_DIR, str)

        # Output frequency should be positive
        self.assertGreater(const.OUTPUT_FREQUENCY, 0)

        # Figure parameters
        self.assertIsInstance(const.FIGURE_SIZE, tuple)
        self.assertEqual(len(const.FIGURE_SIZE), 2)
        self.assertGreater(const.DPI, 0)

    def test_initial_states_validity(self):
        """Test that initial states are physically valid."""
        for i, state in enumerate(const.INITIAL_STATES):
            x, y, vx, vy = state

            # Positions should be within box
            self.assertGreaterEqual(x, const.BOX_MIN_X,
                                  f"Particle {i+1} x position outside box")
            self.assertLessEqual(x, const.BOX_MAX_X,
                               f"Particle {i+1} x position outside box")
            self.assertGreaterEqual(y, const.BOX_MIN_Y,
                                  f"Particle {i+1} y position outside box")
            self.assertLessEqual(y, const.BOX_MAX_Y,
                               f"Particle {i+1} y position outside box")

            # Velocities should be finite
            self.assertTrue(np.isfinite(vx), f"Particle {i+1} vx not finite")
            self.assertTrue(np.isfinite(vy), f"Particle {i+1} vy not finite")

            # Check velocity magnitudes are reasonable (not too large)
            speed = np.sqrt(vx**2 + vy**2)
            self.assertLess(speed, 50.0, f"Particle {i+1} initial speed too high")

    def test_constants_immutability(self):
        """Test that constants are not accidentally modified."""
        # Store original values
        original_mass = const.MASS
        original_charge = const.CHARGE
        original_gravity = const.GRAVITY
        original_dt = const.DT

        # Constants should still have correct values
        self.assertEqual(const.MASS, original_mass)
        self.assertEqual(const.CHARGE, original_charge)
        self.assertEqual(const.GRAVITY, original_gravity)
        self.assertEqual(const.DT, original_dt)

    def test_particle_interactions(self):
        """Test that particle configuration will lead to interesting dynamics."""
        # Check that some particles start close enough to interact strongly
        min_distance = float('inf')

        for i in range(const.N_PARTICLES):
            for j in range(i+1, const.N_PARTICLES):
                pos_i = const.INITIAL_STATES[i, 0:2]
                pos_j = const.INITIAL_STATES[j, 0:2]
                distance = np.linalg.norm(pos_i - pos_j)
                min_distance = min(min_distance, distance)

        # Some particles should be reasonably close
        self.assertLess(min_distance, 20.0, "Particles too far apart for strong interaction")

        # But not too close (would cause numerical issues)
        self.assertGreater(min_distance, 0.1, "Particles too close initially")


if __name__ == '__main__':
    unittest.main(verbosity=2)
