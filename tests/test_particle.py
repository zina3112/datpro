"""
test_particle.py - Unit tests for Particle class

Tests particle properties, state management, and energy calculations.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.particle import Particle
import src.constants as const


class TestParticle(unittest.TestCase):
    """Test suite for Particle class."""

    def setUp(self):
        """Set up test particles."""
        self.particle1 = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0)
        self.particle2 = Particle(x=15.0, y=25.0, vx=-2.0, vy=4.0,
                                 mass=2.0, charge=100.0)

    def test_initialization(self):
        """Test particle initialization with default and custom values."""
        # Test default mass and charge
        self.assertEqual(self.particle1.mass, const.MASS)
        self.assertEqual(self.particle1.charge, const.CHARGE)

        # Test custom mass and charge
        self.assertEqual(self.particle2.mass, 2.0)
        self.assertEqual(self.particle2.charge, 100.0)

        # Test position and velocity
        self.assertEqual(self.particle1.x, 10.0)
        self.assertEqual(self.particle1.y, 20.0)
        self.assertEqual(self.particle1.vx, 5.0)
        self.assertEqual(self.particle1.vy, -3.0)

    def test_state_vector(self):
        """Test state vector representation."""
        expected_state = np.array([10.0, 20.0, 5.0, -3.0])
        np.testing.assert_array_equal(self.particle1.state, expected_state)

        # Test that state is a numpy array
        self.assertIsInstance(self.particle1.state, np.ndarray)
        self.assertEqual(self.particle1.state.dtype, np.float64)

    def test_position_property(self):
        """Test position property getter and setter."""
        # Test getter
        pos = self.particle1.position
        np.testing.assert_array_equal(pos, np.array([10.0, 20.0]))

        # Test setter
        new_pos = np.array([30.0, 40.0])
        self.particle1.position = new_pos
        np.testing.assert_array_equal(self.particle1.position, new_pos)
        self.assertEqual(self.particle1.x, 30.0)
        self.assertEqual(self.particle1.y, 40.0)

    def test_velocity_property(self):
        """Test velocity property getter and setter."""
        # Test getter
        vel = self.particle1.velocity
        np.testing.assert_array_equal(vel, np.array([5.0, -3.0]))

        # Test setter
        new_vel = np.array([7.0, -8.0])
        self.particle1.velocity = new_vel
        np.testing.assert_array_equal(self.particle1.velocity, new_vel)
        self.assertEqual(self.particle1.vx, 7.0)
        self.assertEqual(self.particle1.vy, -8.0)

    def test_kinetic_energy(self):
        """Test kinetic energy calculation."""
        # KE = 0.5 * m * (vx^2 + vy^2)
        # For particle1: 0.5 * 1.0 * (5^2 + 3^2) = 0.5 * 34 = 17.0
        expected_ke = 0.5 * self.particle1.mass * (5.0**2 + 3.0**2)
        self.assertAlmostEqual(self.particle1.kinetic_energy(), expected_ke)

        # For particle2 with mass=2.0
        expected_ke2 = 0.5 * 2.0 * ((-2.0)**2 + 4.0**2)
        self.assertAlmostEqual(self.particle2.kinetic_energy(), expected_ke2)

    def test_gravitational_potential_energy(self):
        """Test gravitational potential energy calculation."""
        # PE = -m * g * y where g is negative
        # For particle1: -1.0 * (-10.0) * 20.0 = 200.0
        expected_pe = -self.particle1.mass * const.GRAVITY * self.particle1.y
        self.assertAlmostEqual(self.particle1.potential_energy_gravity(), expected_pe)

        # Verify it increases with height
        self.particle1.position = np.array([10.0, 30.0])
        new_pe = self.particle1.potential_energy_gravity()
        self.assertGreater(new_pe, expected_pe)

    def test_distance_to(self):
        """Test distance calculation between particles."""
        distance = self.particle1.distance_to(self.particle2)
        # Distance = sqrt((15-10)^2 + (25-20)^2) = sqrt(25 + 25) = sqrt(50)
        expected_distance = np.sqrt(50)
        self.assertAlmostEqual(distance, expected_distance)

        # Test distance is symmetric
        distance_reverse = self.particle2.distance_to(self.particle1)
        self.assertAlmostEqual(distance, distance_reverse)

        # Test zero distance for same particle
        self.assertAlmostEqual(self.particle1.distance_to(self.particle1), 0.0)

    def test_displacement_to(self):
        """Test displacement vector calculation."""
        displacement = self.particle1.displacement_to(self.particle2)
        # Displacement from particle2 to particle1: (10-15, 20-25) = (-5, -5)
        expected = np.array([-5.0, -5.0])
        np.testing.assert_array_almost_equal(displacement, expected)

        # Test reverse displacement
        displacement_reverse = self.particle2.displacement_to(self.particle1)
        np.testing.assert_array_almost_equal(displacement_reverse, -expected)

    def test_update_state(self):
        """Test state update functionality."""
        new_state = np.array([100.0, 200.0, 10.0, -20.0])
        self.particle1.update_state(new_state)

        np.testing.assert_array_equal(self.particle1.state, new_state)
        self.assertEqual(self.particle1.x, 100.0)
        self.assertEqual(self.particle1.y, 200.0)
        self.assertEqual(self.particle1.vx, 10.0)
        self.assertEqual(self.particle1.vy, -20.0)

        # Test invalid state vector
        with self.assertRaises(ValueError):
            self.particle1.update_state(np.array([1.0, 2.0]))  # Wrong size

    def test_copy(self):
        """Test particle deep copy."""
        copy = self.particle1.copy()

        # Test that values are equal
        np.testing.assert_array_equal(copy.state, self.particle1.state)
        self.assertEqual(copy.mass, self.particle1.mass)
        self.assertEqual(copy.charge, self.particle1.charge)

        # Test that it's a deep copy (modifying copy doesn't affect original)
        copy.update_state(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertNotEqual(copy.x, self.particle1.x)

    def test_unique_particle_ids(self):
        """Test that each particle gets a unique ID."""
        p1 = Particle(0, 0, 0, 0)
        p2 = Particle(0, 0, 0, 0)
        p3 = Particle(0, 0, 0, 0)

        # All IDs should be different
        self.assertNotEqual(p1.particle_id, p2.particle_id)
        self.assertNotEqual(p2.particle_id, p3.particle_id)
        self.assertNotEqual(p1.particle_id, p3.particle_id)

    def test_collision_tracking(self):
        """Test collision count tracking."""
        self.assertEqual(self.particle1.collision_count, 0)
        self.assertEqual(self.particle1.last_collision_time, -1.0)

        # Simulate collision
        self.particle1.collision_count += 1
        self.particle1.last_collision_time = 5.5

        self.assertEqual(self.particle1.collision_count, 1)
        self.assertEqual(self.particle1.last_collision_time, 5.5)

    def test_string_representation(self):
        """Test string representations."""
        str_repr = str(self.particle1)
        self.assertIn("Particle", str_repr)
        self.assertIn("10.00", str_repr)  # x position
        self.assertIn("20.00", str_repr)  # y position

        repr_str = repr(self.particle1)
        self.assertIn("Particle(", repr_str)
        self.assertIn("x=10.0", repr_str)


if __name__ == '__main__':
    unittest.main(verbosity=2)
