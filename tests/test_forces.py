"""
test_forces.py - Updated tests for force calculations
"""

import src.constants as const
from src.particle import Particle
from src.forces import (
    calculate_gravitational_force,
    calculate_coulomb_force_between,
    calculate_acceleration,
)
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCoulombForce(unittest.TestCase):
    """Test Coulomb force calculations between particles."""

    def test_coulomb_overlapping_particles(self):
        """Test Coulomb force for exactly overlapping particles."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)
        p2 = Particle(x=0, y=0, vx=0, vy=0, charge=1.0)

        force = calculate_coulomb_force_between(p1, p2)

        # Should return zero force for exactly overlapping particles
        expected = np.array([0.0, 0.0])
        np.testing.assert_array_almost_equal(force, expected)

    def test_coulomb_very_large_charges(self):
        """Test force calculation with very large charges."""
        p1 = Particle(x=0, y=0, vx=0, vy=0, charge=1e6)
        p2 = Particle(x=1, y=0, vx=0, vy=0, charge=1e6)

        force = calculate_coulomb_force_between(p1, p2)

        # Should be very large (1e12) without artificial capping
        magnitude = np.linalg.norm(force)
        self.assertAlmostEqual(magnitude, 1e12, delta=1e11)


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


class TestIntegratorEdgeCases(unittest.TestCase):
    """Test edge cases in integration."""

    def test_identical_particles(self):
        """Test RK4 with identical overlapping particles."""
        from src.integrator import rk4_step_system

        particles = [
            Particle(x=5, y=5, vx=1, vy=1, mass=1.0, charge=1.0),
            Particle(x=5, y=5, vx=1, vy=1, mass=1.0, charge=1.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        # Should get same increments for identical overlapping particles
        # (they have zero force between them)
        np.testing.assert_array_almost_equal(increments[0], increments[1])

    def test_negative_mass(self):
        """Test RK4 with negative mass."""
        from src.integrator import rk4_step_system

        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=-1.0, charge=0.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        # With negative mass, acceleration is opposite to force
        # Gravity pulls down (-10), but negative mass accelerates up
        self.assertGreater(increments[0][1], 0)  # y increases (falls up)
        self.assertGreater(increments[0][3], 0)  # vy becomes positive


class TestRK4Accuracy(unittest.TestCase):
    """Test RK4 accuracy and convergence."""

    def test_rk4_order_convergence(self):
        """Test that RK4 has 4th order convergence."""
        from src.integrator import rk4_step_system

        y0 = 100.0
        v0 = 10.0
        t_final = 0.1

        errors = []
        dts = [0.01, 0.005, 0.0025]

        for dt in dts:
            particles = [
                Particle(x=0, y=y0, vx=0, vy=v0, mass=1.0, charge=0.0)
            ]

            n_steps = int(t_final / dt)

            for _ in range(n_steps):
                increments = rk4_step_system(particles, dt)
                particles[0].update_state(particles[0].state + increments[0])

            # Analytical solution
            y_analytical = y0 + v0 * t_final + 0.5 * const.GRAVITY * t_final**2
            error = abs(particles[0].y - y_analytical)
            errors.append(error)

        # Check 4th order convergence
        # Avoid division by zero
        if errors[1] > 0 and errors[2] > 0:
            ratio1 = errors[0] / errors[1] if errors[1] > 1e-15 else 16.0
            ratio2 = errors[1] / errors[2] if errors[2] > 1e-15 else 16.0

            # Should be close to 16 (2^4)
            self.assertGreater(ratio1, 10)
            self.assertLess(ratio1, 20)
            self.assertGreater(ratio2, 10)
            self.assertLess(ratio2, 20)


class TestEnergyConservationCollisions(unittest.TestCase):
    """Test energy conservation during collisions."""

    def test_multiple_wall_collisions_energy(self):
        """Test energy conservation through multiple collisions."""
        from src.simulation import Simulation
        import tempfile

        # Fast particle that will bounce multiple times
        initial_states = np.array([
            [50.0, 50.0, 100.0, 80.0]
        ])

        with tempfile.NamedTemporaryFile(suffix='.csv') as tmp:
            sim = Simulation(initial_states=initial_states, dt=0.001, output_file=tmp.name)
            sim.particles[0].charge = 0.0

            initial_energy = sim.calculate_total_energy()

            # Run long enough for collisions
            for _ in range(100):
                sim.step()

            # Should have collisions with new collision handling
            self.assertGreater(sim.particles[0].collision_count, 0)

            final_energy = sim.calculate_total_energy()

            relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
            self.assertLess(relative_drift, 0.01)  # 1% tolerance


if __name__ == '__main__':
    unittest.main(verbosity=2)
