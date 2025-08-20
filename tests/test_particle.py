"""
test_particle.py - Comprehensive unit tests for Particle class

Tests all particle functionality including initialization, properties,
energy calculations, and state management.
"""

import src.constants as const
from src.particle import Particle
import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParticleInitialization(unittest.TestCase):
    """Test particle initialization and default values."""

    def test_basic_initialization(self):
        """Test particle creation with explicit values."""
        p = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0, mass=2.0, charge=10.0)

        self.assertEqual(p.x, 10.0)
        self.assertEqual(p.y, 20.0)
        self.assertEqual(p.vx, 5.0)
        self.assertEqual(p.vy, -3.0)
        self.assertEqual(p.mass, 2.0)
        self.assertEqual(p.charge, 10.0)

    def test_default_mass_charge(self):
        """Test particle uses default mass and charge from constants."""
        p = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0)

        self.assertEqual(p.mass, const.MASS)
        self.assertEqual(p.charge, const.CHARGE)

    def test_state_vector_initialization(self):
        """Test state vector is properly constructed."""
        p = Particle(x=1.0, y=2.0, vx=3.0, vy=4.0)

        expected_state = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(p.state, expected_state)

    def test_initial_state_storage(self):
        """Test initial state is stored for reference."""
        p = Particle(x=5.0, y=10.0, vx=1.0, vy=2.0)

        np.testing.assert_array_equal(p.initial_state, p.state)

        # Modify state and check initial is unchanged
        p.state[0] = 99.0
        self.assertEqual(p.initial_state[0], 5.0)

    def test_particle_id_uniqueness(self):
        """Test each particle gets unique ID."""
        particles = [Particle(0, 0, 0, 0) for _ in range(10)]
        ids = [p.particle_id for p in particles]

        # All IDs should be unique
        self.assertEqual(len(ids), len(set(ids)))

    def test_negative_values(self):
        """Test particle handles negative positions and velocities."""
        p = Particle(x=-100.0, y=-200.0, vx=-5.0, vy=-10.0)

        self.assertEqual(p.x, -100.0)
        self.assertEqual(p.y, -200.0)
        self.assertEqual(p.vx, -5.0)
        self.assertEqual(p.vy, -10.0)

    def test_zero_mass_charge(self):
        """Test particle with zero mass and charge."""
        p = Particle(x=0, y=0, vx=0, vy=0, mass=0.0, charge=0.0)

        self.assertEqual(p.mass, 0.0)
        self.assertEqual(p.charge, 0.0)


class TestParticleProperties(unittest.TestCase):
    """Test particle property getters and setters."""

    def setUp(self):
        """Create test particle."""
        self.p = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0)

    def test_position_getter(self):
        """Test position property returns correct array."""
        pos = self.p.position
        np.testing.assert_array_equal(pos, np.array([10.0, 20.0]))

    def test_position_setter(self):
        """Test position can be set via property."""
        new_pos = np.array([30.0, 40.0])
        self.p.position = new_pos

        np.testing.assert_array_equal(self.p.position, new_pos)
        self.assertEqual(self.p.x, 30.0)
        self.assertEqual(self.p.y, 40.0)

    def test_velocity_getter(self):
        """Test velocity property returns correct array."""
        vel = self.p.velocity
        np.testing.assert_array_equal(vel, np.array([5.0, -3.0]))

    def test_velocity_setter(self):
        """Test velocity can be set via property."""
        new_vel = np.array([7.0, -8.0])
        self.p.velocity = new_vel

        np.testing.assert_array_equal(self.p.velocity, new_vel)
        self.assertEqual(self.p.vx, 7.0)
        self.assertEqual(self.p.vy, -8.0)

    def test_individual_coordinate_access(self):
        """Test individual x, y, vx, vy properties."""
        self.assertEqual(self.p.x, 10.0)
        self.assertEqual(self.p.y, 20.0)
        self.assertEqual(self.p.vx, 5.0)
        self.assertEqual(self.p.vy, -3.0)


class TestParticleEnergy(unittest.TestCase):
    """Test energy calculation methods."""

    def test_kinetic_energy_basic(self):
        """Test kinetic energy calculation."""
        p = Particle(x=0, y=0, vx=3.0, vy=4.0, mass=2.0)

        # KE = 0.5 * m * v^2 = 0.5 * 2 * (9 + 16) = 25
        ke = p.kinetic_energy()
        self.assertAlmostEqual(ke, 25.0, places=10)

    def test_kinetic_energy_zero_velocity(self):
        """Test KE for stationary particle."""
        p = Particle(x=0, y=0, vx=0, vy=0, mass=5.0)

        ke = p.kinetic_energy()
        self.assertEqual(ke, 0.0)

    def test_kinetic_energy_negative_velocity(self):
        """Test KE with negative velocities (should still be positive)."""
        p = Particle(x=0, y=0, vx=-3.0, vy=-4.0, mass=2.0)

        ke = p.kinetic_energy()
        self.assertAlmostEqual(ke, 25.0, places=10)

    def test_gravitational_potential_energy(self):
        """Test gravitational PE calculation."""
        p = Particle(x=0, y=50.0, vx=0, vy=0, mass=2.0)

        # PE = -m * g * y = -2 * (-10) * 50 = 1000
        pe = p.potential_energy_gravity()
        self.assertAlmostEqual(pe, 1000.0, places=10)

    def test_gravitational_pe_at_origin(self):
        """Test gravitational PE at y=0."""
        p = Particle(x=0, y=0.0, vx=0, vy=0, mass=1.0)

        pe = p.potential_energy_gravity()
        self.assertEqual(pe, 0.0)

    def test_gravitational_pe_negative_y(self):
        """Test gravitational PE for negative y position."""
        p = Particle(x=0, y=-10.0, vx=0, vy=0, mass=1.0)

        # PE = -1 * (-10) * (-10) = -100
        pe = p.potential_energy_gravity()
        self.assertAlmostEqual(pe, -100.0, places=10)


class TestParticleInteractions(unittest.TestCase):
    """Test particle-to-particle calculations."""

    def test_distance_to_horizontal(self):
        """Test distance calculation along x-axis."""
        p1 = Particle(x=0.0, y=0.0, vx=0, vy=0)
        p2 = Particle(x=5.0, y=0.0, vx=0, vy=0)

        distance = p1.distance_to(p2)
        self.assertAlmostEqual(distance, 5.0, places=10)

    def test_distance_to_vertical(self):
        """Test distance calculation along y-axis."""
        p1 = Particle(x=0.0, y=0.0, vx=0, vy=0)
        p2 = Particle(x=0.0, y=5.0, vx=0, vy=0)

        distance = p1.distance_to(p2)
        self.assertAlmostEqual(distance, 5.0, places=10)

    def test_distance_to_diagonal(self):
        """Test distance for diagonal separation (3-4-5 triangle)."""
        p1 = Particle(x=0.0, y=0.0, vx=0, vy=0)
        p2 = Particle(x=3.0, y=4.0, vx=0, vy=0)

        distance = p1.distance_to(p2)
        self.assertAlmostEqual(distance, 5.0, places=10)

    def test_distance_to_self(self):
        """Test distance to same position is zero."""
        p1 = Particle(x=5.0, y=10.0, vx=0, vy=0)
        p2 = Particle(x=5.0, y=10.0, vx=0, vy=0)

        distance = p1.distance_to(p2)
        self.assertAlmostEqual(distance, 0.0, places=10)

    def test_displacement_to(self):
        """Test displacement vector calculation."""
        p1 = Particle(x=5.0, y=10.0, vx=0, vy=0)
        p2 = Particle(x=2.0, y=6.0, vx=0, vy=0)

        # Displacement from p2 to p1
        displacement = p1.displacement_to(p2)
        expected = np.array([3.0, 4.0])
        np.testing.assert_array_almost_equal(displacement, expected)

    def test_displacement_to_negative(self):
        """Test displacement with negative coordinates."""
        p1 = Particle(x=-5.0, y=-10.0, vx=0, vy=0)
        p2 = Particle(x=5.0, y=10.0, vx=0, vy=0)

        displacement = p1.displacement_to(p2)
        expected = np.array([-10.0, -20.0])
        np.testing.assert_array_almost_equal(displacement, expected)


class TestParticleStateManagement(unittest.TestCase):
    """Test state update and copy operations."""

    def test_update_state_valid(self):
        """Test updating particle state with valid vector."""
        p = Particle(x=0, y=0, vx=0, vy=0)
        new_state = np.array([10.0, 20.0, 5.0, -3.0])

        p.update_state(new_state)

        np.testing.assert_array_equal(p.state, new_state)
        self.assertEqual(p.x, 10.0)
        self.assertEqual(p.y, 20.0)
        self.assertEqual(p.vx, 5.0)
        self.assertEqual(p.vy, -3.0)

    def test_update_state_invalid_size(self):
        """Test update_state rejects wrong size vectors."""
        p = Particle(x=0, y=0, vx=0, vy=0)

        with self.assertRaises(ValueError):
            p.update_state(np.array([1.0, 2.0]))  # Too short

        with self.assertRaises(ValueError):
            p.update_state(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))  # Too long

    def test_copy_deep(self):
        """Test copy creates independent particle."""
        p1 = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0, mass=2.0, charge=10.0)
        p1.collision_count = 5
        p1.last_collision_time = 1.5

        p2 = p1.copy()

        # Check values are equal
        np.testing.assert_array_equal(p2.state, p1.state)
        self.assertEqual(p2.mass, p1.mass)
        self.assertEqual(p2.charge, p1.charge)
        self.assertEqual(p2.collision_count, p1.collision_count)
        self.assertEqual(p2.last_collision_time, p1.last_collision_time)

        # Check they're independent
        p2.state[0] = 999.0
        self.assertEqual(p1.x, 10.0)  # p1 unchanged

    def test_collision_tracking(self):
        """Test collision count and time tracking."""
        p = Particle(x=0, y=0, vx=0, vy=0)

        self.assertEqual(p.collision_count, 0)
        self.assertEqual(p.last_collision_time, -1.0)

        # Simulate collision
        p.collision_count += 1
        p.last_collision_time = 2.5

        self.assertEqual(p.collision_count, 1)
        self.assertEqual(p.last_collision_time, 2.5)


class TestParticleStringRepresentation(unittest.TestCase):
    """Test string representations of particles."""

    def test_str_representation(self):
        """Test human-readable string representation."""
        p = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0, mass=2.0, charge=10.0)
        s = str(p)

        self.assertIn("Particle", s)
        self.assertIn("10.00", s)  # x position
        self.assertIn("20.00", s)  # y position
        self.assertIn("5.00", s)   # vx
        self.assertIn("-3.00", s)  # vy

    def test_repr_representation(self):
        """Test technical representation."""
        p = Particle(x=10.0, y=20.0, vx=5.0, vy=-3.0, mass=2.0, charge=10.0)
        r = repr(p)

        self.assertIn("Particle(", r)
        self.assertIn("x=10", r)
        self.assertIn("y=20", r)
        self.assertIn("vx=5", r)
        self.assertIn("vy=-3", r)
        self.assertIn("mass=2", r)
        self.assertIn("charge=10", r)


class TestParticleEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_very_large_values(self):
        """Test particle with very large coordinate values."""
        p = Particle(x=1e10, y=1e10, vx=1e5, vy=1e5)

        self.assertEqual(p.x, 1e10)
        self.assertEqual(p.y, 1e10)
        ke = p.kinetic_energy()
        self.assertGreater(ke, 0)

    def test_very_small_values(self):
        """Test particle with very small non-zero values."""
        p = Particle(x=1e-10, y=1e-10, vx=1e-10, vy=1e-10)

        self.assertEqual(p.x, 1e-10)
        self.assertEqual(p.y, 1e-10)
        ke = p.kinetic_energy()
        self.assertGreaterEqual(ke, 0)

    def test_zero_mass_kinetic_energy(self):
        """Test KE calculation with zero mass."""
        p = Particle(x=0, y=0, vx=10.0, vy=10.0, mass=0.0)

        ke = p.kinetic_energy()
        self.assertEqual(ke, 0.0)

    def test_state_vector_immutability(self):
        """Test that state updates don't affect other references."""
        p = Particle(x=1, y=2, vx=3, vy=4)
        state_copy = p.state.copy()

        p.update_state(np.array([5, 6, 7, 8]))

        # Original copy should be unchanged
        np.testing.assert_array_equal(state_copy, np.array([1, 2, 3, 4]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
