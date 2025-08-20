"""
test_energy.py - Comprehensive energy conservation tests

Detailed tests for energy conservation under various conditions including
collisions, multi-particle interactions, and different configurations.
"""

import src.constants as const
from src.forces import calculate_potential_energy_coulomb
from src.particle import Particle
from src.simulation import Simulation
import unittest
import numpy as np
import sys
import os
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnergyComponents(unittest.TestCase):
    """Test individual energy component calculations."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_kinetic_energy_calculation(self):
        """Test kinetic energy calculation for various velocities."""
        test_cases = [
            # (vx, vy, mass, expected_ke)
            (0.0, 0.0, 1.0, 0.0),           # Stationary
            (3.0, 4.0, 1.0, 12.5),          # 3-4-5 triangle
            (10.0, 0.0, 2.0, 100.0),        # Horizontal motion
            (0.0, 10.0, 2.0, 100.0),        # Vertical motion
            (-5.0, -5.0, 1.0, 25.0),        # Negative velocities
        ]

        for vx, vy, mass, expected_ke in test_cases:
            p = Particle(x=0, y=0, vx=vx, vy=vy, mass=mass)
            ke = p.kinetic_energy()
            self.assertAlmostEqual(ke, expected_ke, places=10,
                                   msg=f"KE failed for vx={vx}, vy={vy}, m={mass}")

    def test_gravitational_potential_energy(self):
        """Test gravitational PE for various heights."""
        test_cases = [
            # (y, mass, expected_pe)
            (0.0, 1.0, 0.0),              # At origin
            (10.0, 1.0, 100.0),           # Positive height
            (100.0, 2.0, 2000.0),         # Higher with more mass
            (-10.0, 1.0, -100.0),         # Below origin
        ]

        for y, mass, expected_pe in test_cases:
            p = Particle(x=0, y=y, vx=0, vy=0, mass=mass)
            pe = p.potential_energy_gravity()
            # PE = -m * g * y, with g = -10
            self.assertAlmostEqual(pe, expected_pe, places=10,
                                   msg=f"PE failed for y={y}, m={mass}")

    def test_coulomb_potential_energy_pairs(self):
        """Test Coulomb PE for particle pairs."""
        test_cases = [
            # (r, q1, q2, expected_pe)
            (1.0, 1.0, 1.0, 1.0),          # Unit charges at unit distance
            (2.0, 1.0, 1.0, 0.5),          # Double distance
            (1.0, 2.0, 3.0, 6.0),          # Different charges
            (10.0, 10.0, 10.0, 10.0),      # Larger charges
        ]

        for r, q1, q2, expected_pe in test_cases:
            particles = [
                Particle(x=0, y=0, vx=0, vy=0, charge=q1),
                Particle(x=r, y=0, vx=0, vy=0, charge=q2)
            ]

            pe = calculate_potential_energy_coulomb(particles)
            self.assertAlmostEqual(pe, expected_pe, places=5,
                                   msg=f"Coulomb PE failed for r={r}, q1={q1}, q2={q2}")

    def test_total_energy_single_particle(self):
        """Test total energy for single particle system."""
        initial_states = np.array([
            [0.0, 100.0, 10.0, 10.0]  # KE = 100, PE_grav = 1000
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)

        total_energy = sim.calculate_total_energy()

        # KE = 0.5 * 1 * (100 + 100) = 100
        # PE_grav = -1 * (-10) * 100 = 1000
        # No Coulomb for single particle
        expected = 1100.0
        self.assertAlmostEqual(total_energy, expected, places=5)


class TestEnergyConservationStationary(unittest.TestCase):
    """Test energy conservation for stationary configurations."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_stationary_particle_no_forces(self):
        """Test energy conservation for stationary particle with no forces."""
        # Temporarily set gravity and charge to zero
        original_gravity = const.GRAVITY
        const.GRAVITY = 0.0

        try:
            initial_states = np.array([
                [50.0, 50.0, 0.0, 0.0]
            ])

            output_file = os.path.join(self.temp_dir, "test.csv")
            sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
            sim.particles[0].charge = 0.0

            initial_energy = sim.calculate_total_energy()

            # Run simulation
            for _ in range(100):
                sim.step()

            final_energy = sim.calculate_total_energy()

            # Energy should be exactly conserved (no forces)
            self.assertAlmostEqual(final_energy, initial_energy, places=10)

            # Particle should not have moved
            self.assertAlmostEqual(sim.particles[0].x, 50.0, places=10)
            self.assertAlmostEqual(sim.particles[0].y, 50.0, places=10)

        finally:
            const.GRAVITY = original_gravity

    def test_balanced_forces_equilibrium(self):
        """Test energy conservation at force equilibrium."""
        # Two particles at specific distance where forces might balance
        initial_states = np.array([
            [40.0, 50.0, 0.0, 0.0],
            [60.0, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)

        initial_energy = sim.calculate_total_energy()

        # Run for short time with small timestep
        for _ in range(100):
            sim.step()

        final_energy = sim.calculate_total_energy()

        # Energy should be well conserved
        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-6)


class TestEnergyConservationDynamics(unittest.TestCase):
    """Test energy conservation during dynamic motion."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_free_fall_energy_conservation(self):
        """Test energy conservation during free fall."""
        initial_states = np.array([
            [50.0, 80.0, 0.0, 0.0]  # Start high, zero velocity
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0  # No Coulomb force

        initial_energy = sim.calculate_total_energy()

        # Let particle fall for a while
        for _ in range(100):
            sim.step()

        final_energy = sim.calculate_total_energy()

        # Energy should be conserved (PE converts to KE)
        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-5)

        # Particle should have fallen
        self.assertLess(sim.particles[0].y, 80.0)

        # Velocity should be negative (falling)
        self.assertLess(sim.particles[0].vy, 0)

    def test_projectile_motion_energy(self):
        """Test energy conservation for projectile motion."""
        initial_states = np.array([
            [10.0, 50.0, 20.0, 30.0]  # Diagonal launch
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()

        # Run simulation
        for _ in range(50):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-5)

    def test_two_body_orbit_energy(self):
        """Test energy conservation for two-body interaction."""
        # Two particles with perpendicular velocities
        initial_states = np.array([
            [30.0, 50.0, 0.0, 10.0],
            [70.0, 50.0, 0.0, -10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)

        initial_energy = sim.calculate_total_energy()

        # Run for many steps
        for _ in range(500):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-4)


class TestEnergyConservationCollisions(unittest.TestCase):
    """Test energy conservation during wall collisions."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_single_wall_collision_energy(self):
        """Test energy conservation for single wall collision."""
        # Particle heading toward right wall
        initial_states = np.array([
            [95.0, 50.0, 50.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()

        # Run until collision happens
        for _ in range(200):
            sim.step()

        # Verify collision occurred
        self.assertGreater(sim.particles[0].collision_count, 0)

        final_energy = sim.calculate_total_energy()

        # Energy should be conserved through collision
        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-4)

        # Velocity should be reversed
        self.assertLess(sim.particles[0].vx, 0)

    def test_multiple_wall_collisions_energy(self):
        """Test energy conservation through multiple collisions."""
        # Fast particle that will bounce multiple times
        initial_states = np.array([
            [50.0, 50.0, 100.0, 80.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()

        # Run long enough for multiple collisions
        for _ in range(500):
            sim.step()

        # Should have multiple collisions
        self.assertGreater(sim.particles[0].collision_count, 2)

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-3)

    def test_corner_collision_energy(self):
        """Test energy conservation for corner collision."""
        # Particle heading toward corner
        initial_states = np.array([
            [95.0, 95.0, 50.0, 50.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()

        # Run until collision
        for _ in range(200):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-3)


class TestEnergyConservationMultiParticle(unittest.TestCase):
    """Test energy conservation with multiple particles."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_three_particle_system_energy(self):
        """Test energy conservation for three-particle system."""
        initial_states = np.array([
            [30.0, 50.0, 10.0, 0.0],
            [50.0, 50.0, -10.0, 10.0],
            [70.0, 50.0, 0.0, -10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)

        initial_energy = sim.calculate_total_energy()

        # Run simulation
        for _ in range(200):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-4)

    def test_standard_seven_particle_energy(self):
        """Test energy conservation for standard 7-particle configuration."""
        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(dt=0.001, output_file=output_file)  # Uses default 7 particles

        initial_energy = sim.calculate_total_energy()

        # Expected initial energy from problem
        self.assertAlmostEqual(initial_energy, 6658.979, places=2)

        # Run for 0.1 seconds
        for _ in range(100):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-4)

    def test_symmetric_configuration_energy(self):
        """Test energy conservation for symmetric particle configuration."""
        # Four particles in square formation
        initial_states = np.array([
            [40.0, 40.0, 5.0, 5.0],
            [60.0, 40.0, -5.0, 5.0],
            [60.0, 60.0, -5.0, -5.0],
            [40.0, 60.0, 5.0, -5.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)

        initial_energy = sim.calculate_total_energy()

        # Run simulation
        for _ in range(200):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-4)


class TestEnergyDriftAnalysis(unittest.TestCase):
    """Test energy drift behavior under various conditions."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_energy_drift_vs_timestep(self):
        """Test that energy drift decreases with smaller timestep."""
        initial_states = np.array([
            [30.0, 50.0, 10.0, 10.0],
            [70.0, 50.0, -10.0, -10.0]
        ])

        timesteps = [0.01, 0.001, 0.0001]
        drifts = []

        for dt in timesteps:
            output_file = os.path.join(self.temp_dir, f"test_dt_{dt}.csv")
            sim = Simulation(initial_states=initial_states, dt=dt, output_file=output_file)

            initial_energy = sim.calculate_total_energy()

            # Run for same physical time
            n_steps = int(0.1 / dt)
            for _ in range(n_steps):
                sim.step()

            final_energy = sim.calculate_total_energy()
            drift = abs(final_energy - initial_energy) / abs(initial_energy)
            drifts.append(drift)

        # Drift should decrease with smaller timestep
        for i in range(len(drifts) - 1):
            if drifts[i] > 1e-14:  # Skip if already at machine precision
                self.assertLess(drifts[i + 1], drifts[i])

    def test_long_term_energy_stability(self):
        """Test energy stability over long simulation time."""
        initial_states = np.array([
            [50.0, 50.0, 5.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()
        max_drift = 0.0

        # Run for extended time, checking drift periodically
        for i in range(10):
            for _ in range(100):
                sim.step()

            current_energy = sim.calculate_total_energy()
            drift = abs(current_energy - initial_energy) / abs(initial_energy)
            max_drift = max(max_drift, drift)

        # Drift should remain bounded
        self.assertLess(max_drift, 1e-3)

    def test_energy_fluctuations_bounded(self):
        """Test that energy fluctuations remain bounded."""
        initial_states = np.array([
            [20.0, 50.0, 20.0, 0.0],
            [80.0, 50.0, -20.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.0001, output_file=output_file)

        initial_energy = sim.calculate_total_energy()
        energies = [initial_energy]

        # Collect energy values
        for _ in range(100):
            sim.step()
            energies.append(sim.calculate_total_energy())

        # Calculate statistics
        energies = np.array(energies)
        mean_energy = np.mean(energies)
        std_energy = np.std(energies)

        # Standard deviation should be small relative to mean
        relative_std = std_energy / abs(mean_energy)
        self.assertLess(relative_std, 1e-4)


class TestEnergyEdgeCases(unittest.TestCase):
    """Test energy conservation in edge cases."""

    def setUp(self):
        """Create temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_zero_mass_energy(self):
        """Test energy with zero mass particle."""
        initial_states = np.array([
            [50.0, 50.0, 10.0, 10.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].mass = 0.0

        energy = sim.calculate_total_energy()

        # Should have no kinetic or gravitational energy
        self.assertEqual(energy, 0.0)

    def test_zero_charge_energy(self):
        """Test energy with zero charge particles."""
        initial_states = np.array([
            [30.0, 50.0, 10.0, 0.0],
            [70.0, 50.0, -10.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.001, output_file=output_file)
        sim.particles[0].charge = 0.0
        sim.particles[1].charge = 0.0

        energy = sim.calculate_total_energy()

        # Should have only kinetic and gravitational energy
        expected_ke = 0.5 * (100 + 100)
        expected_pe = -const.GRAVITY * (50 + 50)
        expected_total = expected_ke + expected_pe

        self.assertAlmostEqual(energy, expected_total, places=5)

    def test_very_close_particles_energy(self):
        """Test energy with very close particles."""
        initial_states = np.array([
            [50.0, 50.0, 0.0, 0.0],
            [50.0001, 50.0, 0.0, 0.0]
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.00001, output_file=output_file)

        energy = sim.calculate_total_energy()

        # Energy should be finite (regularized)
        self.assertTrue(np.isfinite(energy))
        self.assertLess(energy, 1e10)

    def test_high_speed_energy_conservation(self):
        """Test energy conservation at high speeds."""
        initial_states = np.array([
            [50.0, 50.0, 1000.0, 1000.0]  # Very high speed
        ])

        output_file = os.path.join(self.temp_dir, "test.csv")
        sim = Simulation(initial_states=initial_states, dt=0.00001, output_file=output_file)
        sim.particles[0].charge = 0.0

        initial_energy = sim.calculate_total_energy()

        # Run for very short time with tiny timestep
        for _ in range(10):
            sim.step()

        final_energy = sim.calculate_total_energy()

        relative_drift = abs(final_energy - initial_energy) / abs(initial_energy)
        self.assertLess(relative_drift, 1e-3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
