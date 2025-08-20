"""
test_forces.py - Comprehensive unit tests for force calculations

Tests all force calculation functions including gravity, Coulomb forces,
system forces, and potential energy calculations.
"""

import src.constants as const
from src.forces import (
    calculate_gravitational_force,
    calculate_coulomb_force_between,
    calculate_total_electrostatic_force,
    calculate_total_force,
    calculate_acceleration,
    calculate_potential_energy_coulomb,
    calculate_system_forces,
    calculate_system_forces_symmetric,
    calculate_system_accelerations
)
from src.particle import Particle
import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestGravitationalForce(unittest.TestCase):
    """Test gravitational force calculations."""

    def test_gravity_standard_mass(self):
        """Test gravitational force with unit mass."""
        p = Particle(x=0, y=0, vx=0, vy=0, mass=1.0)
        force = calculate_gravitational_force(p)

        expected = np.array([0.0, const.GRAVITY])
        np.testing.assert_array_almost_equal(force, expected)

    def test_gravity_different_mass(self):
        """Test gravitational force scales with mass."""
        p = Particle(x=0, y=0, vx=0, vy=0, mass=2.5)
        force = calculate_gravitational_force(p)

        expected = np.array([0.0, 2.5 * const.GRAVITY])
        np.testing.assert_array_almost_equal(force, expected)

    def test_gravity_zero_mass(self):
        """Test gravitational force with zero mass."""
        p = Particle(x=0, y=0, vx=0, vy=0, mass=0.0)
        force = calculate_gravitational_force(p)

        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(force, expected)

    def test_gravity_independent_of_position(self):
        """Test gravity doesn't depend on position."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, mass=1.0)
        p2 = Particle(x=100, y=200, vx=0, vy=0, mass=1.0)

        force1 = calculate_gravitational_force(p1)
        force2 = calculate_gravitational_force(p2)

        np.testing.assert_array_almost_equal(force1, force2)

    def test_gravity_independent_of_velocity(self):
        """Test gravity doesn't depend on velocity."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, mass=1.0)
        p2 = Particle(x=0, y=0, vx=100, vy=-50, mass=1.0)

        force1 = calculate_gravitational_force(p1)
        force2 = calculate_gravitational_force(p2)

        np.testing.assert_array_almost_equal(force1, force2)


class TestCoulombForce(unittest.TestCase):
    """Test Coulomb force calculations between particles."""

    def test_coulomb_unit_charges_unit_distance(self):
        """Test Coulomb force for unit charges at unit distance."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=1, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Force should be repulsive (negative x direction for p1)
        self.assertLess(force[0], 0)
        self.assertAlmostEqual(force[1], 0.0, places=10)

        # Magnitude should be approximately 1.0
        magnitude = np.linalg.norm(force)
        self.assertAlmostEqual(magnitude, 1.0, places=2)

    def test_coulomb_diagonal_configuration(self):
        """Test Coulomb force for diagonal particle arrangement."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=2.0)
        p2 = Particle(x=3, y=4, vx=0, vy=0, charge=3.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Distance is 5 (3-4-5 triangle)
        # Force magnitude = 2*3/25 = 0.24
        expected_magnitude = 6.0 / 25.0
        actual_magnitude = np.linalg.norm(force)
        self.assertAlmostEqual(actual_magnitude, expected_magnitude, places=5)

        # Direction should be from p2 to p1 (negative displacement)
        self.assertLess(force[0], 0)  # Negative x component
        self.assertLess(force[1], 0)  # Negative y component

    def test_coulomb_overlapping_particles(self):
        """Test Coulomb force for exactly overlapping particles."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Should return zero force for exactly overlapping
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(force, expected)

    def test_coulomb_very_close_particles(self):
        """Test Coulomb force for very close but not overlapping particles."""
        epsilon = const.EPSILON * 10  # Close but not exactly overlapping
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=epsilon, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Force should be finite (regularized)
        magnitude = np.linalg.norm(force)
        self.assertLess(magnitude, 1e10)  # Not infinite
        self.assertGreater(magnitude, 0)   # Not zero

    def test_coulomb_opposite_charges(self):
        """Test attraction for opposite charges (if charges can be negative)."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=1, y=0, vx=0, vy=0, charge=-1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Force should be attractive (positive x direction for p1)
        self.assertGreater(force[0], 0)

    def test_coulomb_zero_charge(self):
        """Test Coulomb force with zero charge."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=0.0)
        p2 = Particle(x=1, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(force, expected)

    def test_coulomb_symmetry(self):
        """Test Newton's third law for Coulomb forces."""
        p1 = Particle(x=1, y=2, vx=0, vy=0, charge=2.0)
        p2 = Particle(x=4, y=6, vx=0, vy=0, charge=3.0)

        force_on_1 = calculate_coulomb_force_between(p1, p2)
        force_on_2 = calculate_coulomb_force_between(p2, p1)

        # Forces should be equal and opposite
        np.testing.assert_array_almost_equal(force_on_1, -force_on_2)

    def test_coulomb_large_distance(self):
        """Test Coulomb force at large distances (should be very small)."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=1000, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Force should be very small
        magnitude = np.linalg.norm(force)
        self.assertLess(magnitude, 1e-5)


class TestTotalForces(unittest.TestCase):
    """Test total force calculations combining multiple contributions."""

    def test_total_electrostatic_two_particles(self):
        """Test total electrostatic force from two other particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=1.0),  # Target
            Particle(x=1, y=0, vx=0, vy=0, charge=1.0),  # Right
            Particle(x=0, y=1, vx=0, vy=0, charge=1.0),  # Above
        ]

        force = calculate_total_electrostatic_force(0, particles)

        # Force from particle 1: approximately [-1, 0]
        # Force from particle 2: approximately [0, -1]
        # Total: approximately [-1, -1]
        self.assertLess(force[0], 0)
        self.assertLess(force[1], 0)

    def test_total_electrostatic_symmetric_configuration(self):
        """Test total force in symmetric configuration should cancel."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=1.0),   # Center
            Particle(x=1, y=0, vx=0, vy=0, charge=1.0),   # Right
            Particle(x=-1, y=0, vx=0, vy=0, charge=1.0),  # Left
            Particle(x=0, y=1, vx=0, vy=0, charge=1.0),   # Above
            Particle(x=0, y=-1, vx=0, vy=0, charge=1.0),  # Below
        ]

        force = calculate_total_electrostatic_force(0, particles)

        # Forces should cancel due to symmetry
        np.testing.assert_array_almost_equal(force, np.array([0.0, 0.0]), decimal=5)

    def test_total_force_gravity_plus_coulomb(self):
        """Test combined gravitational and Coulomb forces."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
        ]

        force = calculate_total_force(0, particles)

        # Should have gravity (negative y) and Coulomb repulsion (negative x)
        self.assertLess(force[0], 0)  # Coulomb repulsion
        self.assertAlmostEqual(force[1], const.GRAVITY, places=5)

    def test_total_force_single_particle(self):
        """Test total force on single particle (only gravity)."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
        ]

        force = calculate_total_force(0, particles)

        # Only gravity should act
        expected = np.array([0.0, const.GRAVITY])
        np.testing.assert_array_almost_equal(force, expected)


class TestAccelerations(unittest.TestCase):
    """Test acceleration calculations from forces."""

    def test_acceleration_from_force(self):
        """Test acceleration calculation using Newton's second law."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=2.0, charge=0.0),
        ]

        acceleration = calculate_acceleration(0, particles)

        # a = F/m = [0, -10]/2 = [0, -5]
        expected = np.array([0.0, const.GRAVITY / 2.0])
        np.testing.assert_array_almost_equal(acceleration, expected)

    def test_acceleration_zero_mass(self):
        """Test acceleration with zero mass returns zero."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=0.0, charge=1.0),
        ]

        acceleration = calculate_acceleration(0, particles)

        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(acceleration, expected)

    def test_system_accelerations(self):
        """Test acceleration calculation for entire system."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=2.0, charge=1.0),
        ]

        accelerations = calculate_system_accelerations(particles)

        self.assertEqual(len(accelerations), 2)

        # Each should have gravity component
        for acc in accelerations:
            self.assertLess(acc[1], 0)  # Negative y acceleration due to gravity


class TestPotentialEnergy(unittest.TestCase):
    """Test potential energy calculations."""

    def test_coulomb_potential_two_particles(self):
        """Test Coulomb potential energy for two particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=2.0),
            Particle(x=3, y=0, vx=0, vy=0, charge=3.0),
        ]

        potential = calculate_potential_energy_coulomb(particles)

        # U = q1*q2/r = 2*3/3 = 2.0
        self.assertAlmostEqual(potential, 2.0, places=5)

    def test_coulomb_potential_three_particles(self):
        """Test Coulomb potential for three particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=2.0),
            Particle(x=3, y=0, vx=0, vy=0, charge=3.0),
            Particle(x=0, y=4, vx=0, vy=0, charge=1.0),
        ]

        # U = sum of all pairs
        # U_01 = 2*3/3 = 2.0
        # U_02 = 2*1/4 = 0.5
        # U_12 = 3*1/5 = 0.6
        # Total = 3.1

        potential = calculate_potential_energy_coulomb(particles)
        self.assertAlmostEqual(potential, 3.1, places=5)

    def test_coulomb_potential_zero_charge(self):
        """Test potential energy with zero charge particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=0.0),
            Particle(x=1, y=0, vx=0, vy=0, charge=1.0),
        ]

        potential = calculate_potential_energy_coulomb(particles)
        self.assertEqual(potential, 0.0)

    def test_coulomb_potential_overlapping(self):
        """Test potential energy for overlapping particles (regularized)."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=1.0),
            Particle(x=0, y=0, vx=0, vy=0, charge=1.0),
        ]

        potential = calculate_potential_energy_coulomb(particles)

        # Should be finite due to regularization
        self.assertLess(potential, 1e10)
        self.assertGreater(potential, 0)

    def test_coulomb_potential_single_particle(self):
        """Test potential energy for single particle is zero."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=10.0),
        ]

        potential = calculate_potential_energy_coulomb(particles)
        self.assertEqual(potential, 0.0)


class TestSystemForces(unittest.TestCase):
    """Test system-wide force calculations."""

    def test_system_forces_basic(self):
        """Test basic system force calculation."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
        ]

        forces = calculate_system_forces(particles)

        self.assertEqual(len(forces), 2)

        # Particles should repel each other
        self.assertLess(forces[0][0], 0)  # Particle 0 pushed left
        self.assertGreater(forces[1][0], 0)  # Particle 1 pushed right

    def test_system_forces_symmetric(self):
        """Test symmetric force calculation ensures Newton's 3rd law."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=2.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=3.0),
        ]

        forces = calculate_system_forces_symmetric(particles)

        # Remove gravity component for comparison
        coulomb_0 = forces[0] - np.array([0, const.GRAVITY])
        coulomb_1 = forces[1] - np.array([0, const.GRAVITY])

        # Coulomb forces should be equal and opposite
        np.testing.assert_array_almost_equal(coulomb_0, -coulomb_1)

    def test_system_forces_many_particles(self):
        """Test system forces with many particles."""
        # Create 10 particles in a line
        particles = [
            Particle(x=float(i), y=0, vx=0, vy=0, mass=1.0, charge=1.0)
            for i in range(10)
        ]

        forces = calculate_system_forces(particles)

        self.assertEqual(len(forces), 10)

        # First particle should be pushed left
        self.assertLess(forces[0][0], 0)

        # Last particle should be pushed right
        self.assertGreater(forces[-1][0], 0)

    def test_system_forces_consistency(self):
        """Test that regular and symmetric calculations give similar results."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=1, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=2, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
        ]

        forces_regular = calculate_system_forces(particles)
        forces_symmetric = calculate_system_forces_symmetric(particles)

        # Should give same results within numerical precision
        for f1, f2 in zip(forces_regular, forces_symmetric):
            np.testing.assert_array_almost_equal(f1, f2, decimal=10)


class TestForceEdgeCases(unittest.TestCase):
    """Test edge cases in force calculations."""

    def test_very_large_charges(self):
        """Test force calculation with very large charges."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1e6)
        p2 = Particle(x=1, y=0, vx=0, vy=0, charge=1e6)

        force = calculate_coulomb_force_between(p1, p2)

        # Should be very large but finite
        magnitude = np.linalg.norm(force)
        self.assertGreater(magnitude, 1e10)
        self.assertLess(magnitude, np.inf)

    def test_very_small_separation(self):
        """Test forces at very small separations."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=1e-15, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Should be regularized
        magnitude = np.linalg.norm(force)
        self.assertLess(magnitude, 1e20)

    def test_mixed_zero_nonzero_charges(self):
        """Test system with mix of charged and uncharged particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, charge=0.0),
            Particle(x=1, y=0, vx=0, vy=0, charge=1.0),
            Particle(x=2, y=0, vx=0, vy=0, charge=0.0),
            Particle(x=3, y=0, vx=0, vy=0, charge=1.0),
        ]

        # Uncharged particles should only feel gravity
        force_0 = calculate_total_force(0, particles)
        np.testing.assert_array_almost_equal(force_0, [0.0, const.GRAVITY])

        # Charged particles feel both
        force_1 = calculate_total_force(1, particles)
        self.assertNotEqual(force_1[0], 0.0)  # Has Coulomb component


if __name__ == '__main__':
    unittest.main(verbosity=2)
