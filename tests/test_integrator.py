"""
test_integrator.py - Unit tests for RK4 integration

Tests the Runge-Kutta 4th order integration method for accuracy and stability.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.particle import Particle
from src.integrator import (
    rk4_step_single,
    rk4_step_system,
    RK4Integrator,
    state_derivative
)
import src.constants as const


class TestIntegrator(unittest.TestCase):
    """Test suite for RK4 integration."""

    def setUp(self):
        """Set up test particles and integrator."""
        # Simple two-particle system
        self.particles = [
            Particle(x=10.0, y=50.0, vx=5.0, vy=0.0),
            Particle(x=90.0, y=50.0, vx=-5.0, vy=0.0)
        ]

        self.integrator = RK4Integrator(dt=0.001)

    def test_state_derivative_structure(self):
        """Test that state derivative has correct structure."""
        derivatives = state_derivative(self.particles)

        # Should return list of derivatives for each particle
        self.assertEqual(len(derivatives), len(self.particles))

        for deriv in derivatives:
            # Each derivative should be 4D: [vx, vy, ax, ay]
            self.assertEqual(len(deriv), 4)
            self.assertIsInstance(deriv, np.ndarray)

    def test_state_derivative_velocity_components(self):
        """Test that velocity components are correctly placed in derivative."""
        derivatives = state_derivative(self.particles)

        for i, (particle, deriv) in enumerate(zip(self.particles, derivatives)):
            # First two components should be velocity
            np.testing.assert_array_almost_equal(deriv[0:2], particle.velocity)

    def test_rk4_single_particle_free_fall(self):
        """Test RK4 integration for a particle in free fall."""
        # Single particle falling under gravity
        particle = Particle(x=50.0, y=80.0, vx=0.0, vy=0.0, mass=1.0, charge=0.0)
        particles = [particle]
        dt = 0.01

        # Calculate one RK4 step
        increment = rk4_step_single(particle, particles, 0, dt)

        # After small time dt, particle should move slightly down
        # y_new ≈ y + vy*dt + 0.5*g*dt^2
        # With vy=0 initially: Δy ≈ 0.5*g*dt^2
        expected_dy = 0.5 * const.GRAVITY * dt**2

        # Check y displacement (increment[1] is Δy)
        self.assertLess(increment[1], 0)  # Should be negative (falling)
        self.assertAlmostEqual(increment[1], expected_dy, places=4)

    def test_rk4_system_conservation(self):
        """Test that RK4 preserves certain conservation laws."""
        # Two particles with no charge (only gravity)
        particles = [
            Particle(x=30.0, y=50.0, vx=10.0, vy=0.0, charge=0.0),
            Particle(x=70.0, y=50.0, vx=-10.0, vy=0.0, charge=0.0)
        ]

        # Calculate center of mass velocity
        total_momentum_x = sum(p.mass * p.vx for p in particles)
        total_mass = sum(p.mass for p in particles)
        com_vx = total_momentum_x / total_mass

        # Do one RK4 step
        increments = rk4_step_system(particles, 0.001)

        # Apply increments
        for p, inc in zip(particles, increments):
            p.update_state(p.state + inc)

        # Check momentum conservation in x (no horizontal forces)
        new_momentum_x = sum(p.mass * p.vx for p in particles)
        self.assertAlmostEqual(new_momentum_x, total_momentum_x, places=10)

    def test_rk4_accuracy_harmonic_oscillator(self):
        """Test RK4 accuracy using a simple harmonic oscillator analog."""
        # Create a particle attached to a "spring" (using modified forces)
        # This tests the integration accuracy for a known solution

        # We'll simulate a particle with a restoring force
        # For this test, we'll use a single particle and check energy conservation
        particle = Particle(x=55.0, y=50.0, vx=0.0, vy=10.0)
        particles = [particle]

        initial_energy = particle.kinetic_energy() + particle.potential_energy_gravity()

        # Run for multiple steps
        dt = 0.001
        for _ in range(100):
            increment = rk4_step_single(particle, particles, 0, dt)
            particle.update_state(particle.state + increment)

        final_energy = particle.kinetic_energy() + particle.potential_energy_gravity()

        # Energy shouldn't change dramatically (some drift is expected)
        energy_change = abs(final_energy - initial_energy)
        relative_change = energy_change / abs(initial_energy) if initial_energy != 0 else 0

        # RK4 should maintain energy to within 0.1% for this short simulation
        self.assertLess(relative_change, 0.001)

    def test_rk4_timestep_consistency(self):
        """Test that smaller timesteps give more accurate results."""
        # Set up identical initial conditions
        particles_coarse = [
            Particle(x=50.0, y=50.0, vx=10.0, vy=10.0)
        ]
        particles_fine = [
            Particle(x=50.0, y=50.0, vx=10.0, vy=10.0)
        ]

        # Integrate with different timesteps
        dt_coarse = 0.01
        dt_fine = 0.001
        total_time = 0.1

        # Coarse integration
        steps_coarse = int(total_time / dt_coarse)
        for _ in range(steps_coarse):
            inc = rk4_step_system(particles_coarse, dt_coarse)
            particles_coarse[0].update_state(particles_coarse[0].state + inc[0])

        # Fine integration
        steps_fine = int(total_time / dt_fine)
        for _ in range(steps_fine):
            inc = rk4_step_system(particles_fine, dt_fine)
            particles_fine[0].update_state(particles_fine[0].state + inc[0])

        # Positions should be similar but fine should be more accurate
        pos_diff = np.linalg.norm(particles_fine[0].position - particles_coarse[0].position)

        # There should be some difference (coarse is less accurate)
        self.assertGreater(pos_diff, 0.0)
        # But not too much for this simple case
        self.assertLess(pos_diff, 1.0)

    def test_rk4_integrator_class(self):
        """Test the RK4Integrator class functionality."""
        integrator = RK4Integrator(dt=0.001)

        # Test initialization
        self.assertEqual(integrator.dt, 0.001)
        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.total_time, 0.0)

        # Test step method
        particles = [Particle(x=50.0, y=50.0, vx=0.0, vy=0.0)]
        increments = integrator.step(particles)

        self.assertEqual(len(increments), 1)
        self.assertEqual(integrator.step_count, 1)
        self.assertAlmostEqual(integrator.total_time, 0.001)

        # Test reset
        integrator.reset()
        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.total_time, 0.0)

    def test_rk4_with_strong_repulsion(self):
        """Test RK4 stability with strong Coulomb repulsion."""
        # Two highly charged particles close together
        particles = [
            Particle(x=50.0, y=50.0, vx=0.0, vy=0.0, charge=50.0),
            Particle(x=51.0, y=50.0, vx=0.0, vy=0.0, charge=50.0)
        ]

        # Should handle strong repulsion without NaN or infinity
        increments = rk4_step_system(particles, 0.001)

        for inc in increments:
            self.assertTrue(np.isfinite(inc).all())
            # Particles should move apart
            self.assertNotEqual(np.linalg.norm(inc), 0.0)

    def test_rk4_zero_timestep(self):
        """Test that zero timestep returns zero increment."""
        particle = Particle(x=50.0, y=50.0, vx=10.0, vy=10.0)
        particles = [particle]

        increment = rk4_step_single(particle, particles, 0, dt=0.0)
        np.testing.assert_array_almost_equal(increment, np.zeros(4))

    def test_rk4_coefficients(self):
        """Test that RK4 uses correct coefficient weights."""
        # This is implicitly tested by accuracy, but we can verify the structure
        # by checking that the method is 4th order accurate

        # For a linear ODE, RK4 should be exact up to machine precision
        # Test with particle under constant force (gravity only, no Coulomb)
        particle = Particle(x=50.0, y=100.0, vx=0.0, vy=0.0, charge=0.0)
        particles = [particle]

        dt = 0.01
        increment = rk4_step_single(particle, particles, 0, dt)

        # For constant acceleration, exact solution is:
        # Δy = vy*dt + 0.5*ay*dt^2
        # Δvy = ay*dt
        ay = const.GRAVITY  # Acceleration due to gravity
        expected_dvy = ay * dt
        expected_dy = 0.0 * dt + 0.5 * ay * dt**2

        # RK4 should get this exactly right for constant acceleration
        self.assertAlmostEqual(increment[3], expected_dvy, places=10)
        # Y displacement is more complex due to RK4's averaging
        # but should be very close to the analytical solution
        self.assertAlmostEqual(increment[1], expected_dy, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
