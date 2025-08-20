"""
test_integrator.py - Comprehensive unit tests for RK4 integrator

Tests the Runge-Kutta 4th order integration method, state derivatives,
accuracy, and system-wide integration.
"""

import src.constants as const
from src.integrator import (
    state_derivative,
    rk4_step_single,
    rk4_step_system,
    RK4Integrator
)
from src.particle import Particle
import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestStateDerivative(unittest.TestCase):
    """Test state derivative calculations."""

    def test_state_derivative_single_particle_gravity(self):
        """Test derivative for single particle under gravity."""
        particles = [
            Particle(x=0, y=10, vx=5.0, vy=3.0, mass=1.0, charge=0.0)
        ]

        derivatives = state_derivative(particles)

        # Expected: [vx, vy, ax, ay] = [5.0, 3.0, 0.0, -10.0]
        expected = np.array([5.0, 3.0, 0.0, const.GRAVITY])
        np.testing.assert_array_almost_equal(derivatives[0], expected)

    def test_state_derivative_no_forces(self):
        """Test derivative with no forces (zero mass, zero charge)."""
        particles = [
            Particle(x=0, y=10, vx=5.0, vy=3.0, mass=0.0, charge=0.0)
        ]

        derivatives = state_derivative(particles)

        # Expected: [vx, vy, ax, ay] = [5.0, 3.0, 0.0, 0.0]
        expected = np.array([5.0, 3.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(derivatives[0], expected)

    def test_state_derivative_two_particles(self):
        """Test derivative for two interacting particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=1.0)
        ]

        derivatives = state_derivative(particles)

        self.assertEqual(len(derivatives), 2)

        # Particle 0 should have negative x acceleration (repulsion)
        self.assertLess(derivatives[0][2], 0)

        # Particle 1 should have positive x acceleration (repulsion)
        self.assertGreater(derivatives[1][2], 0)

        # Both should have negative y acceleration (gravity)
        self.assertAlmostEqual(derivatives[0][3], const.GRAVITY)
        self.assertAlmostEqual(derivatives[1][3], const.GRAVITY)

    def test_state_derivative_stationary_particles(self):
        """Test derivative for stationary particles."""
        particles = [
            Particle(x=5, y=10, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        derivatives = state_derivative(particles)

        # Velocity components should be zero
        self.assertEqual(derivatives[0][0], 0.0)
        self.assertEqual(derivatives[0][1], 0.0)

        # Only gravity acceleration
        self.assertEqual(derivatives[0][2], 0.0)
        self.assertEqual(derivatives[0][3], const.GRAVITY)


class TestRK4StepSingle(unittest.TestCase):
    """Test single particle RK4 integration."""

    def test_rk4_single_free_fall(self):
        """Test RK4 step for free falling particle."""
        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        dt = 0.01
        increment = rk4_step_single(particles[0], particles, 0, dt)

        # Should move down (negative y increment)
        self.assertEqual(increment[0], 0.0)  # No x motion
        self.assertLess(increment[1], 0)      # Falls down
        self.assertEqual(increment[2], 0.0)  # No x acceleration
        self.assertLess(increment[3], 0)      # Velocity becomes negative

    def test_rk4_single_projectile(self):
        """Test RK4 step for projectile motion."""
        particles = [
            Particle(x=0, y=50, vx=10, vy=10, mass=1.0, charge=0.0)
        ]

        dt = 0.01
        increment = rk4_step_single(particles[0], particles, 0, dt)

        # Horizontal motion should be uniform
        self.assertAlmostEqual(increment[0], 10 * dt, places=5)

        # Vertical motion should be affected by gravity
        self.assertGreater(increment[1], 0)  # Still moving up initially
        self.assertLess(increment[3], 0)     # But velocity decreasing

    def test_rk4_single_with_coulomb(self):
        """Test RK4 step with Coulomb forces."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=1.0)
        ]

        dt = 0.01
        increment = rk4_step_single(particles[0], particles, 0, dt)

        # Should be repelled (negative x increment)
        self.assertLess(increment[0], 0)
        self.assertLess(increment[2], 0)  # Negative x velocity change

    def test_rk4_single_consistency(self):
        """Test that single particle RK4 is self-consistent."""
        particles = [
            Particle(x=5, y=10, vx=2, vy=3, mass=1.0, charge=0.0)
        ]

        dt = 0.001

        # Two half steps
        inc1 = rk4_step_single(particles[0], particles, 0, dt / 2)
        particles[0].update_state(particles[0].state + inc1)
        inc2 = rk4_step_single(particles[0], particles, 0, dt / 2)

        # Reset and do one full step
        particles[0] = Particle(x=5, y=10, vx=2, vy=3, mass=1.0, charge=0.0)
        inc_full = rk4_step_single(particles[0], particles, 0, dt)

        # Should be very close (RK4 is 4th order accurate)
        # Note: Not exactly equal due to nonlinearity
        total_two_steps = inc1 + inc2
        np.testing.assert_array_almost_equal(total_two_steps, inc_full, decimal=3)


class TestRK4StepSystem(unittest.TestCase):
    """Test system-wide RK4 integration."""

    def test_rk4_system_single_particle(self):
        """Test system RK4 with single particle."""
        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        self.assertEqual(len(increments), 1)

        # Should fall under gravity
        self.assertLess(increments[0][1], 0)  # y decreases
        self.assertLess(increments[0][3], 0)  # vy becomes negative

    def test_rk4_system_two_particles(self):
        """Test system RK4 with two particles."""
        particles = [
            Particle(x=0, y=0, vx=0, vy=0, mass=1.0, charge=1.0),
            Particle(x=1, y=0, vx=0, vy=0, mass=1.0, charge=1.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        self.assertEqual(len(increments), 2)

        # Particles should repel
        self.assertLess(increments[0][0], 0)  # Particle 0 moves left
        self.assertGreater(increments[1][0], 0)  # Particle 1 moves right

    def test_rk4_system_many_particles(self):
        """Test system RK4 with many particles."""
        n = 10
        particles = [
            Particle(x=float(i), y=0, vx=0, vy=0, mass=1.0, charge=0.1)
            for i in range(n)
        ]

        dt = 0.001
        increments = rk4_step_system(particles, dt)

        self.assertEqual(len(increments), n)

        # First particle should move left
        self.assertLess(increments[0][0], 0)

        # Last particle should move right
        self.assertGreater(increments[-1][0], 0)

    def test_rk4_system_preserves_center_of_mass(self):
        """Test that RK4 preserves center of mass without external forces."""
        # Two particles with equal mass, no gravity for this test
        original_gravity = const.GRAVITY
        const.GRAVITY = 0.0

        try:
            particles = [
                Particle(x=0, y=0, vx=5, vy=0, mass=1.0, charge=0.0),
                Particle(x=10, y=0, vx=-5, vy=0, mass=1.0, charge=0.0)
            ]

            # Calculate initial center of mass velocity
            initial_vcm = (particles[0].vx + particles[1].vx) / 2

            dt = 0.01
            increments = rk4_step_system(particles, dt)

            # Update particles
            for p, inc in zip(particles, increments):
                p.update_state(p.state + inc)

            # Check center of mass velocity unchanged
            final_vcm = (particles[0].vx + particles[1].vx) / 2
            self.assertAlmostEqual(initial_vcm, final_vcm, places=10)

        finally:
            const.GRAVITY = original_gravity


class TestRK4Integrator(unittest.TestCase):
    """Test RK4Integrator class."""

    def test_integrator_initialization(self):
        """Test integrator initialization."""
        integrator = RK4Integrator(dt=0.01)

        self.assertEqual(integrator.dt, 0.01)
        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.total_time, 0.0)

    def test_integrator_step(self):
        """Test single integration step."""
        integrator = RK4Integrator(dt=0.01)

        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        increments = integrator.step(particles)

        self.assertEqual(integrator.step_count, 1)
        self.assertAlmostEqual(integrator.total_time, 0.01)
        self.assertEqual(len(increments), 1)

    def test_integrator_multiple_steps(self):
        """Test multiple integration steps."""
        integrator = RK4Integrator(dt=0.01)

        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        for i in range(10):
            increments = integrator.step(particles)
            for p, inc in zip(particles, increments):
                p.update_state(p.state + inc)

        self.assertEqual(integrator.step_count, 10)
        self.assertAlmostEqual(integrator.total_time, 0.1)

        # Particle should have fallen
        self.assertLess(particles[0].y, 100)

    def test_integrator_integrate_to_time(self):
        """Test integration to specific time."""
        integrator = RK4Integrator(dt=0.01)

        particles = [
            Particle(x=0, y=100, vx=10, vy=0, mass=1.0, charge=0.0)
        ]

        target_time = 0.5
        integrator.integrate_to_time(particles, target_time)

        self.assertAlmostEqual(integrator.total_time, target_time)
        self.assertEqual(integrator.step_count, 50)

        # Particle should have moved horizontally
        self.assertAlmostEqual(particles[0].x, 5.0, places=2)

        # And fallen due to gravity
        self.assertLess(particles[0].y, 100)

    def test_integrator_callback(self):
        """Test integration with callback function."""
        integrator = RK4Integrator(dt=0.01)

        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        callback_times = []

        def callback(time, particles):
            callback_times.append(time)

        integrator.integrate_to_time(particles, 0.1, callback=callback)

        self.assertEqual(len(callback_times), 10)
        self.assertAlmostEqual(callback_times[-1], 0.1)

    def test_integrator_reset(self):
        """Test integrator reset."""
        integrator = RK4Integrator(dt=0.01)

        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        # Do some steps
        for _ in range(5):
            integrator.step(particles)

        self.assertEqual(integrator.step_count, 5)
        self.assertAlmostEqual(integrator.total_time, 0.05)

        # Reset
        integrator.reset()

        self.assertEqual(integrator.step_count, 0)
        self.assertEqual(integrator.total_time, 0.0)


class TestRK4Accuracy(unittest.TestCase):
    """Test RK4 accuracy and convergence."""

    def test_rk4_accuracy_free_fall(self):
        """Test RK4 accuracy for analytical solution (free fall)."""
        y0 = 100.0
        v0 = 0.0

        particles = [
            Particle(x=0, y=y0, vx=0, vy=v0, mass=1.0, charge=0.0)
        ]

        dt = 0.001
        n_steps = 100
        t_final = n_steps * dt

        for _ in range(n_steps):
            increments = rk4_step_system(particles, dt)
            particles[0].update_state(particles[0].state + increments[0])

        # Analytical solution: y = y0 + v0*t + 0.5*g*t^2
        y_analytical = y0 + v0 * t_final + 0.5 * const.GRAVITY * t_final**2

        # RK4 should be very accurate
        self.assertAlmostEqual(particles[0].y, y_analytical, places=4)

    def test_rk4_order_convergence(self):
        """Test that RK4 has 4th order convergence."""
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

        # Check 4th order convergence: error ~ dt^4
        # When dt is halved, error should decrease by factor of 16
        ratio1 = errors[0] / errors[1]
        ratio2 = errors[1] / errors[2]

        # Should be close to 16 (2^4)
        self.assertGreater(ratio1, 10)
        self.assertLess(ratio1, 20)
        self.assertGreater(ratio2, 10)
        self.assertLess(ratio2, 20)

    def test_rk4_energy_conservation(self):
        """Test energy conservation with RK4."""
        # Two particles with Coulomb interaction
        particles = [
            Particle(x=10, y=50, vx=5, vy=0, mass=1.0, charge=10.0),
            Particle(x=90, y=50, vx=-5, vy=0, mass=1.0, charge=10.0)
        ]

        def calculate_energy(particles):
            total_e = 0
            # Kinetic
            for p in particles:
                total_e += 0.5 * p.mass * (p.vx**2 + p.vy**2)
            # Gravitational
            for p in particles:
                total_e += -p.mass * const.GRAVITY * p.y
            # Coulomb
            if len(particles) > 1:
                r = particles[0].distance_to(particles[1])
                if r > const.EPSILON * 100:
                    total_e += particles[0].charge * particles[1].charge / r
            return total_e

        initial_energy = calculate_energy(particles)

        # Integrate
        dt = 0.0001
        for _ in range(100):
            increments = rk4_step_system(particles, dt)
            for p, inc in zip(particles, increments):
                p.update_state(p.state + inc)

        final_energy = calculate_energy(particles)

        # Energy should be conserved to high precision
        relative_error = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_error, 1e-6)


class TestIntegratorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_timestep(self):
        """Test RK4 with zero timestep."""
        particles = [
            Particle(x=5, y=10, vx=2, vy=3, mass=1.0, charge=1.0)
        ]

        dt = 0.0
        increments = rk4_step_system(particles, dt)

        # Should be zero increment
        np.testing.assert_array_almost_equal(increments[0], np.zeros(4))

    def test_very_small_timestep(self):
        """Test RK4 with very small timestep."""
        particles = [
            Particle(x=0, y=100, vx=10, vy=0, mass=1.0, charge=0.0)
        ]

        dt = 1e-10
        increments = rk4_step_system(particles, dt)

        # Increments should be proportional to dt
        self.assertLess(np.linalg.norm(increments[0]), 1e-8)

    def test_very_large_timestep(self):
        """Test RK4 stability with large timestep."""
        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=1.0, charge=0.0)
        ]

        dt = 10.0  # Very large
        increments = rk4_step_system(particles, dt)

        # Should still give finite results
        self.assertTrue(np.all(np.isfinite(increments[0])))

    def test_zero_mass_particle(self):
        """Test RK4 with zero mass particle."""
        particles = [
            Particle(x=0, y=100, vx=10, vy=10, mass=0.0, charge=0.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        # Should have uniform motion (no acceleration)
        self.assertAlmostEqual(increments[0][0], 10 * dt)  # x += vx * dt
        self.assertAlmostEqual(increments[0][1], 10 * dt)  # y += vy * dt
        self.assertEqual(increments[0][2], 0.0)  # No vx change
        self.assertEqual(increments[0][3], 0.0)  # No vy change

    def test_identical_particles(self):
        """Test RK4 with identical overlapping particles."""
        particles = [
            Particle(x=5, y=5, vx=1, vy=1, mass=1.0, charge=1.0),
            Particle(x=5, y=5, vx=1, vy=1, mass=1.0, charge=1.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        # Should get same increments (symmetric situation)
        np.testing.assert_array_almost_equal(increments[0], increments[1])

    def test_negative_mass(self):
        """Test RK4 with negative mass (unphysical but should handle)."""
        particles = [
            Particle(x=0, y=100, vx=0, vy=0, mass=-1.0, charge=0.0)
        ]

        dt = 0.01
        increments = rk4_step_system(particles, dt)

        # Should "fall" upward with negative mass
        self.assertGreater(increments[0][1], 0)  # y increases
        self.assertGreater(increments[0][3], 0)  # vy becomes positive


if __name__ == '__main__':
    unittest.main(verbosity=2)
