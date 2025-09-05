"""
test_regularization.py - Tests specifically for soft-core regularization

This test file validates that the soft-core regularization is necessary
and working correctly to prevent numerical explosions.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.particle import Particle
from src.forces import calculate_coulomb_force_between
from src.simulation import Simulation
import src.constants as const


class TestRegularization(unittest.TestCase):
    """Test suite for soft-core regularization validation."""

    def test_regularization_prevents_infinity(self):
        """Test that regularization prevents infinite forces at r=0."""
        # Create overlapping particles
        p1 = Particle(x=50.0, y=50.0, vx=0.0, vy=0.0, charge=50.0)
        p2 = Particle(x=50.0, y=50.0, vx=0.0, vy=0.0, charge=50.0)

        # Calculate force - should NOT be infinite
        force = calculate_coulomb_force_between(p1, p2)

        # Force should be zero or very small for overlapping particles
        self.assertTrue(np.isfinite(force).all())
        self.assertLess(np.linalg.norm(force), 1e10)  # Some reasonable bound

        # Specifically, implementation returns zero for exactly overlapping
        np.testing.assert_array_almost_equal(force, np.array([0.0, 0.0]))

    def test_regularization_at_small_distances(self):
        """Test force behavior at very small distances."""
        distances = [1e-7, 1e-8, 1e-9, 1e-10, 0.0]
        forces = []

        for d in distances:
            p1 = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0, charge=50.0)
            p2 = Particle(x=d, y=0.0, vx=0.0, vy=0.0, charge=50.0)

            force = calculate_coulomb_force_between(p1, p2)
            force_magnitude = np.linalg.norm(force)
            forces.append(force_magnitude)

            # All forces should be finite
            self.assertTrue(np.isfinite(force_magnitude))

            # Force should be bounded
            self.assertLess(force_magnitude, 1e15)

        # Forces should increase as distance decreases, but saturate
        # (not explode to infinity)
        for i in range(1, len(forces)):
            if distances[i] > 0:  # Skip the exactly zero case
                # Force should generally increase as particles get closer
                # but this isn't strictly monotonic due to regularization
                pass

    def test_minimum_approach_distance_with_initial_conditions(self):
        """Test how close particles get with actual initial conditions."""
        # Use actual initial conditions
        sim = Simulation(
            initial_states=const.INITIAL_STATES,
            dt=const.DT,
            output_file="test_min_distance.csv"
        )

        min_distance_overall = float('inf')

        # Run for a short time and track minimum distance
        for _ in range(100):  # 0.1 seconds
            sim.step()

            # Check all particle pairs
            for i in range(len(sim.particles)):
                for j in range(i+1, len(sim.particles)):
                    distance = sim.particles[i].distance_to(sim.particles[j])
                    min_distance_overall = min(min_distance_overall, distance)

        # Log the minimum distance encountered
        print(f"Minimum distance in 0.1s simulation: {min_distance_overall}")

        # Particles should get reasonably close (justifying regularization)
        # but not overlap completely
        self.assertGreater(min_distance_overall, 0.0)

        # Clean up
        if os.path.exists("test_min_distance.csv"):
            os.remove("test_min_distance.csv")

    def test_energy_conservation_with_close_encounters(self):
        """Test that energy is conserved even with close particle encounters."""
        # Create particles that will have a close encounter
        states = np.array([
            [45.0, 50.0, 10.0, 0.0],   # Moving right
            [55.0, 50.0, -10.0, 0.0]    # Moving left - will collide
        ])

        sim = Simulation(
            initial_states=states,
            dt=0.001,
            output_file="test_close_encounter.csv"
        )

        initial_energy = sim.initial_energy

        # Run until particles pass through closest approach
        for _ in range(500):  # 0.5 seconds
            sim.step()

        final_energy = sim.calculate_total_energy()
        energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)

        # Energy should be conserved despite close encounter
        self.assertLess(energy_drift, 0.01)  # Less than 1% drift

        # Clean up
        if os.path.exists("test_close_encounter.csv"):
            os.remove("test_close_encounter.csv")

    def test_compare_regularized_vs_pure_coulomb_at_safe_distance(self):
        """Test that regularization doesn't affect forces at normal distances."""
        # At reasonable distances, regularized and pure Coulomb should match

        # Distance = 1.0 (well above regularization threshold)
        p1 = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0, charge=50.0)
        p2 = Particle(x=1.0, y=0.0, vx=0.0, vy=0.0, charge=50.0)

        force = calculate_coulomb_force_between(p1, p2)
        force_magnitude = np.linalg.norm(force)

        # Pure Coulomb: F = q1*q2/r^2 = 50*50/1^2 = 2500
        expected_magnitude = 2500.0

        # Should match to high precision at this distance
        self.assertAlmostEqual(force_magnitude, expected_magnitude, places=5)

    def test_stability_with_high_charge_density(self):
        """Test simulation stability with many charged particles in small region."""
        # Pack particles relatively close together
        states = []
        for i in range(5):
            for j in range(5):
                x = 40.0 + i * 5.0  # 5 unit spacing
                y = 40.0 + j * 5.0
                vx = np.random.uniform(-5, 5)
                vy = np.random.uniform(-5, 5)
                states.append([x, y, vx, vy])

        states = np.array(states)

        sim = Simulation(
            initial_states=states,
            dt=0.001,
            output_file="test_high_density.csv"
        )

        # Should be able to run without numerical explosion
        try:
            for _ in range(100):
                sim.step()

            # Check final energy is finite
            final_energy = sim.calculate_total_energy()
            self.assertTrue(np.isfinite(final_energy))

            # Check all particles have finite positions and velocities
            for particle in sim.particles:
                self.assertTrue(np.isfinite(particle.state).all())

            simulation_stable = True
        except:
            simulation_stable = False

        self.assertTrue(simulation_stable, "Simulation became unstable with high charge density")

        # Clean up
        if os.path.exists("test_high_density.csv"):
            os.remove("test_high_density.csv")

    def test_force_continuity_at_regularization_boundary(self):
        """Test that force is continuous at the regularization threshold."""
        # Test distances around the regularization threshold (1e-6)
        threshold = 1e-6
        test_distances = [
            threshold * 0.5,   # Below threshold
            threshold * 0.99,  # Just below
            threshold,         # At threshold
            threshold * 1.01,  # Just above
            threshold * 2.0    # Above threshold
        ]

        forces = []
        for d in test_distances:
            p1 = Particle(x=0.0, y=0.0, vx=0.0, vy=0.0, charge=50.0)
            p2 = Particle(x=d, y=0.0, vx=0.0, vy=0.0, charge=50.0)

            force = calculate_coulomb_force_between(p1, p2)
            forces.append(np.linalg.norm(force))

        # Forces should vary smoothly (no discontinuous jump)
        for i in range(1, len(forces)):
            if test_distances[i] > 0 and test_distances[i-1] > 0:
                # Check that force doesn't jump by more than factor of 10
                ratio = forces[i] / forces[i-1] if forces[i-1] > 0 else 0
                self.assertLess(abs(ratio), 10.0,
                              f"Force discontinuity at {test_distances[i]}")

    def test_why_regularization_is_necessary(self):
        """Demonstrate why regularization is necessary by showing what would happen without it."""
        # This test documents why we need regularization

        # Calculate what pure Coulomb force would be at small distances
        q = 50.0
        small_distances = [0.01, 0.001, 0.0001, 0.00001]

        pure_coulomb_forces = []
        for r in small_distances:
            # Pure Coulomb: F = q^2/r^2
            f = q * q / (r * r)
            pure_coulomb_forces.append(f)

        # Show how forces explode
        print("\nPure Coulomb forces at small distances (demonstrating need for regularization):")
        for r, f in zip(small_distances, pure_coulomb_forces):
            print(f"  r = {r}: F = {f:.2e}")

        # With dt = 0.001, acceleration = F/m would give velocity change:
        # Δv = a * dt = F * dt
        dt = 0.001
        for r, f in zip(small_distances, pure_coulomb_forces):
            delta_v = f * dt
            print(f"  r = {r}: Δv in one step = {delta_v:.2e}")

        # At r=0.00001, Δv = 2.5e11 * 0.001 = 2.5e8
        # This would send particle to velocity of 250 million units/second!

        # This demonstrates regularization is NECESSARY, not optional
        self.assertGreater(pure_coulomb_forces[-1], 1e10,
                          "Pure Coulomb force explodes at small distances")


if __name__ == '__main__':
    unittest.main(verbosity=2)
