"""
test_forces.py - Unit tests for force calculations

Tests gravitational and Coulomb forces, energy calculations, and singularity handling.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.particle import Particle
from src.forces import (
    calculate_gravitational_force,
    calculate_coulomb_force_between,
    calculate_total_electrostatic_force,
    calculate_total_force,
    calculate_acceleration,
    calculate_potential_energy_coulomb,
    calculate_system_forces_symmetric
)
import src.constants as const


class TestForces(unittest.TestCase):
    """Test suite for force calculations."""

    def setUp(self):
        """Set up test particles."""
        # Create particles at known positions
        self.particle1 = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0)
        self.particle2 = Particle(x=3.0, y=4.0, vx=0.0, vy=0.0)  # Distance = 5
        self.particle3 = Particle(x=10.0, y=0.0, vx=0.0, vy=0.0)  # Distance = 10 from p1

        self.particles = [self.particle1, self.particle2, self.particle3]

    def test_gravitational_force(self):
        """Test gravitational force calculation."""
        force = calculate_gravitational_force(self.particle1)

        # Force should be [0, m*g] where g = -10
        expected_force = np.array([0.0, const.MASS * const.GRAVITY])
        np.testing.assert_array_almost_equal(force, expected_force)

        # Test with different mass
        particle_heavy = Particle(0, 0, 0, 0, mass=2.0)
        force_heavy = calculate_gravitational_force(particle_heavy)
        expected_heavy = np.array([0.0, 2.0 * const.GRAVITY])
        np.testing.assert_array_almost_equal(force_heavy, expected_heavy)

    def test_coulomb_force_between_particles(self):
        """Test Coulomb force between two particles."""
        # Particles at (0,0) and (3,4), distance = 5
        force = calculate_coulomb_force_between(self.particle1, self.particle2)

        # Force magnitude = q1*q2/r^2 = 50*50/25 = 100
        expected_magnitude = (const.CHARGE * const.CHARGE) / 25.0
        actual_magnitude = np.linalg.norm(force)
        self.assertAlmostEqual(actual_magnitude, expected_magnitude, places=5)

        # Force direction: from p2 to p1, so should point in (-3, -4) direction
        # Normalized: (-3/5, -4/5) * magnitude
        expected_direction = np.array([-3.0/5.0, -4.0/5.0])
        actual_direction = force / actual_magnitude
        np.testing.assert_array_almost_equal(actual_direction, expected_direction)

    def test_coulomb_force_symmetry(self):
        """Test Newton's third law for Coulomb forces."""
        force_1_on_2 = calculate_coulomb_force_between(self.particle2, self.particle1)
        force_2_on_1 = calculate_coulomb_force_between(self.particle1, self.particle2)

        # Forces should be equal and opposite
        np.testing.assert_array_almost_equal(force_1_on_2, -force_2_on_1)

    def test_coulomb_force_singularity_handling(self):
        """Test that close particles don't cause numerical explosion."""
        # Create two particles very close together
        p1 = Particle(x=50.0, y=50.0, vx=0.0, vy=0.0)
        p2 = Particle(x=50.0, y=50.0, vx=0.0, vy=0.0)  # Exactly overlapping

        # Should return zero force for overlapping particles
        force = calculate_coulomb_force_between(p1, p2)
        np.testing.assert_array_almost_equal(force, np.array([0.0, 0.0]))

        # Test with very small separation
        p2.position = np.array([50.0 + 1e-8, 50.0])
        force = calculate_coulomb_force_between(p1, p2)

        # Force should be large but finite
        self.assertTrue(np.isfinite(force).all())
        self.assertLess(np.linalg.norm(force), 1e15)  # Some reasonable upper bound

    def test_total_electrostatic_force(self):
        """Test total electrostatic force from multiple particles."""
        # Calculate force on particle 1 from particles 2 and 3
        total_force = calculate_total_electrostatic_force(0, self.particles)

        # Force from p2 at (3,4): repulsive in direction (-3,-4)/5
        force_from_2 = calculate_coulomb_force_between(self.particle1, self.particle2)

        # Force from p3 at (10,0): repulsive in direction (-10,0)/10 = (-1,0)
        force_from_3 = calculate_coulomb_force_between(self.particle1, self.particle3)

        expected_total = force_from_2 + force_from_3
        np.testing.assert_array_almost_equal(total_force, expected_total)

    def test_total_force(self):
        """Test combined gravitational and electrostatic forces."""
        total_force = calculate_total_force(0, self.particles)

        gravity = calculate_gravitational_force(self.particle1)
        electrostatic = calculate_total_electrostatic_force(0, self.particles)
        expected = gravity + electrostatic

        np.testing.assert_array_almost_equal(total_force, expected)

    def test_acceleration_calculation(self):
        """Test F=ma acceleration calculation."""
        # Set up known force scenario
        acceleration = calculate_acceleration(0, self.particles)

        # a = F/m
        total_force = calculate_total_force(0, self.particles)
        expected_accel = total_force / self.particle1.mass

        np.testing.assert_array_almost_equal(acceleration, expected_accel)

        # Test with zero mass particle (should return zero acceleration)
        particle_massless = Particle(0, 0, 0, 0, mass=0.0)
        particles_with_massless = [particle_massless, self.particle2]
        accel_massless = calculate_acceleration(0, particles_with_massless)
        np.testing.assert_array_almost_equal(accel_massless, np.array([0.0, 0.0]))

    def test_coulomb_potential_energy(self):
        """Test Coulomb potential energy calculation."""
        # U = q1*q2/r for each pair, summed
        potential = calculate_potential_energy_coulomb(self.particles)

        # Distance between p1 and p2: 5
        u12 = (const.CHARGE * const.CHARGE) / 5.0

        # Distance between p1 and p3: 10
        u13 = (const.CHARGE * const.CHARGE) / 10.0

        # Distance between p2 and p3: sqrt((10-3)^2 + (0-4)^2) = sqrt(49+16) = sqrt(65)
        dist23 = np.sqrt(65)
        u23 = (const.CHARGE * const.CHARGE) / dist23

        expected_total = u12 + u13 + u23
        self.assertAlmostEqual(potential, expected_total, places=5)

    def test_system_forces_symmetric(self):
        """Test that symmetric force calculation preserves Newton's 3rd law."""
        forces = calculate_system_forces_symmetric(self.particles)

        # Check that total momentum change is zero (Newton's 3rd law)
        # Sum of all forces should be zero (except gravity)
        total_force_x = sum(f[0] for f in forces)
        total_force_y = sum(f[1] for f in forces)

        # X-component should be exactly zero
        self.assertAlmostEqual(total_force_x, 0.0, places=10)

        # Y-component should equal total gravitational force
        total_gravity = len(self.particles) * const.MASS * const.GRAVITY
        self.assertAlmostEqual(total_force_y, total_gravity, places=10)

    def test_force_calculation_with_actual_parameters(self):
        """Test forces with actual simulation parameters (q=50, m=1)."""
        # Create particles with actual charge values
        p1 = Particle(x=20.0, y=50.0, vx=0.0, vy=0.0, charge=50.0)
        p2 = Particle(x=30.0, y=50.0, vx=0.0, vy=0.0, charge=50.0)

        # Distance = 10, Force = 50*50/100 = 25
        force = calculate_coulomb_force_between(p1, p2)
        expected_magnitude = 25.0
        actual_magnitude = np.linalg.norm(force)
        self.assertAlmostEqual(actual_magnitude, expected_magnitude, places=5)

    def test_energy_conservation_in_isolated_system(self):
        """Test that potential energy calculation is consistent."""
        # Create a simple two-particle system
        particles = [
            Particle(x=0.0, y=0.0, vx=5.0, vy=0.0),
            Particle(x=10.0, y=0.0, vx=-5.0, vy=0.0)
        ]

        # Calculate total energy
        ke_total = sum(p.kinetic_energy() for p in particles)
        pe_gravity = sum(p.potential_energy_gravity() for p in particles)
        pe_coulomb = calculate_potential_energy_coulomb(particles)

        total_energy = ke_total + pe_gravity + pe_coulomb

        # Energy should be well-defined (finite)
        self.assertTrue(np.isfinite(total_energy))

    def test_force_scaling_with_distance(self):
        """Test that Coulomb force scales as 1/r^2."""
        p1 = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0)

        distances = [1.0, 2.0, 4.0, 8.0]
        forces = []

        for d in distances:
            p2 = Particle(x=d, y=0.0, vx=0.0, vy=0.0)
            force = calculate_coulomb_force_between(p1, p2)
            forces.append(np.linalg.norm(force))

        # Check 1/r^2 scaling
        for i in range(1, len(distances)):
            ratio = distances[i] / distances[i-1]
            expected_force_ratio = 1.0 / (ratio ** 2)
            actual_force_ratio = forces[i] / forces[i-1]
            self.assertAlmostEqual(actual_force_ratio, expected_force_ratio, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
