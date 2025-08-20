"""
test_constants.py - Comprehensive unit tests for constants module

Tests all constants definitions, values, and consistency checks.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.constants as const


class TestPhysicalConstants(unittest.TestCase):
    """Test physical constants definitions."""

    def test_mass_constant(self):
        """Test mass constant is defined and positive."""
        self.assertIsNotNone(const.MASS)
        self.assertGreater(const.MASS, 0)
        self.assertEqual(const.MASS, 1.0)

    def test_charge_constant(self):
        """Test charge constant is defined and positive."""
        self.assertIsNotNone(const.CHARGE)
        self.assertGreater(const.CHARGE, 0)
        self.assertEqual(const.CHARGE, 50.0)

    def test_gravity_constant(self):
        """Test gravity constant is defined and negative."""
        self.assertIsNotNone(const.GRAVITY)
        self.assertLess(const.GRAVITY, 0)
        self.assertEqual(const.GRAVITY, -10.0)

    def test_constants_type(self):
        """Test that constants are numeric types."""
        self.assertIsInstance(const.MASS, (int, float))
        self.assertIsInstance(const.CHARGE, (int, float))
        self.assertIsInstance(const.GRAVITY, (int, float))


class TestBoxParameters(unittest.TestCase):
    """Test box dimension parameters."""

    def test_box_boundaries_defined(self):
        """Test all box boundaries are defined."""
        self.assertIsNotNone(const.BOX_MIN_X)
        self.assertIsNotNone(const.BOX_MAX_X)
        self.assertIsNotNone(const.BOX_MIN_Y)
        self.assertIsNotNone(const.BOX_MAX_Y)

    def test_box_dimensions_valid(self):
        """Test box dimensions are valid (max > min)."""
        self.assertGreater(const.BOX_MAX_X, const.BOX_MIN_X)
        self.assertGreater(const.BOX_MAX_Y, const.BOX_MIN_Y)

    def test_box_size_calculation(self):
        """Test box width and height calculations."""
        self.assertEqual(const.BOX_WIDTH, const.BOX_MAX_X - const.BOX_MIN_X)
        self.assertEqual(const.BOX_HEIGHT, const.BOX_MAX_Y - const.BOX_MIN_Y)

    def test_box_standard_values(self):
        """Test box has expected standard values."""
        self.assertEqual(const.BOX_MIN_X, 0.0)
        self.assertEqual(const.BOX_MAX_X, 100.0)
        self.assertEqual(const.BOX_MIN_Y, 0.0)
        self.assertEqual(const.BOX_MAX_Y, 100.0)
        self.assertEqual(const.BOX_WIDTH, 100.0)
        self.assertEqual(const.BOX_HEIGHT, 100.0)

    def test_box_dimensions_positive(self):
        """Test box dimensions are positive."""
        self.assertGreater(const.BOX_WIDTH, 0)
        self.assertGreater(const.BOX_HEIGHT, 0)


class TestNumericalParameters(unittest.TestCase):
    """Test numerical simulation parameters."""

    def test_timestep_defined(self):
        """Test timestep is defined and positive."""
        self.assertIsNotNone(const.DT)
        self.assertGreater(const.DT, 0)
        self.assertEqual(const.DT, 0.001)

    def test_simulation_time_defined(self):
        """Test simulation time is defined and positive."""
        self.assertIsNotNone(const.SIMULATION_TIME)
        self.assertGreater(const.SIMULATION_TIME, 0)
        self.assertEqual(const.SIMULATION_TIME, 10.0)

    def test_number_of_steps_calculation(self):
        """Test number of steps is calculated correctly."""
        expected_steps = int(const.SIMULATION_TIME / const.DT)
        self.assertEqual(const.N_STEPS, expected_steps)
        self.assertEqual(const.N_STEPS, 10000)

    def test_epsilon_values(self):
        """Test epsilon values for numerical comparisons."""
        self.assertIsNotNone(const.EPSILON)
        self.assertGreater(const.EPSILON, 0)
        self.assertLess(const.EPSILON, 1e-8)
        self.assertEqual(const.EPSILON, 1e-10)

    def test_energy_tolerance(self):
        """Test energy conservation tolerance."""
        self.assertIsNotNone(const.ENERGY_TOLERANCE)
        self.assertGreater(const.ENERGY_TOLERANCE, 0)
        self.assertLess(const.ENERGY_TOLERANCE, 0.01)
        self.assertEqual(const.ENERGY_TOLERANCE, 1e-6)

    def test_collision_parameters(self):
        """Test collision detection parameters."""
        self.assertIsNotNone(const.COLLISION_EPSILON)
        self.assertEqual(const.COLLISION_EPSILON, const.EPSILON)

        self.assertIsNotNone(const.MAX_COLLISION_ITERATIONS)
        self.assertGreater(const.MAX_COLLISION_ITERATIONS, 0)
        self.assertEqual(const.MAX_COLLISION_ITERATIONS, 10)


class TestInitialConditions(unittest.TestCase):
    """Test initial particle conditions."""

    def test_initial_states_defined(self):
        """Test initial states array is defined."""
        self.assertIsNotNone(const.INITIAL_STATES)
        self.assertIsInstance(const.INITIAL_STATES, np.ndarray)

    def test_initial_states_shape(self):
        """Test initial states have correct shape."""
        # Should be (N_particles, 4) for [x, y, vx, vy]
        self.assertEqual(const.INITIAL_STATES.shape[1], 4)
        self.assertEqual(len(const.INITIAL_STATES), 7)

    def test_initial_states_values(self):
        """Test initial states have expected values."""
        expected = np.array([
            [1.0, 45.0, 10.0, 0.0],
            [99.0, 55.0, -10.0, 0.0],
            [10.0, 50.0, 15.0, -15.0],
            [20.0, 30.0, -15.0, -15.0],
            [80.0, 70.0, 15.0, 15.0],
            [80.0, 60.0, 15.0, 15.0],
            [80.0, 50.0, 15.0, 15.0]
        ])

        np.testing.assert_array_equal(const.INITIAL_STATES, expected)

    def test_number_of_particles(self):
        """Test N_PARTICLES matches initial states."""
        self.assertEqual(const.N_PARTICLES, len(const.INITIAL_STATES))
        self.assertEqual(const.N_PARTICLES, 7)

    def test_initial_positions_in_box(self):
        """Test all initial positions are inside the box."""
        for state in const.INITIAL_STATES:
            x, y = state[0], state[1]
            self.assertGreaterEqual(x, const.BOX_MIN_X)
            self.assertLessEqual(x, const.BOX_MAX_X)
            self.assertGreaterEqual(y, const.BOX_MIN_Y)
            self.assertLessEqual(y, const.BOX_MAX_Y)

    def test_initial_states_finite(self):
        """Test all initial values are finite."""
        self.assertTrue(np.all(np.isfinite(const.INITIAL_STATES)))


class TestOutputParameters(unittest.TestCase):
    """Test output and file parameters."""

    def test_output_directory_defined(self):
        """Test output directory is defined."""
        self.assertIsNotNone(const.OUTPUT_DIR)
        self.assertIsInstance(const.OUTPUT_DIR, str)
        self.assertEqual(const.OUTPUT_DIR, "output")

    def test_output_file_defined(self):
        """Test output file name is defined."""
        self.assertIsNotNone(const.OUTPUT_FILE)
        self.assertIsInstance(const.OUTPUT_FILE, str)
        self.assertTrue(const.OUTPUT_FILE.endswith('.csv'))
        self.assertEqual(const.OUTPUT_FILE, "simulation_results.csv")

    def test_plot_directory_defined(self):
        """Test plot directory is defined."""
        self.assertIsNotNone(const.PLOT_DIR)
        self.assertIsInstance(const.PLOT_DIR, str)
        self.assertEqual(const.PLOT_DIR, "plots")

    def test_output_frequency(self):
        """Test output frequency parameter."""
        self.assertIsNotNone(const.OUTPUT_FREQUENCY)
        self.assertGreater(const.OUTPUT_FREQUENCY, 0)
        self.assertEqual(const.OUTPUT_FREQUENCY, 1)

    def test_figure_parameters(self):
        """Test figure size and DPI parameters."""
        self.assertIsNotNone(const.FIGURE_SIZE)
        self.assertIsInstance(const.FIGURE_SIZE, tuple)
        self.assertEqual(len(const.FIGURE_SIZE), 2)
        self.assertEqual(const.FIGURE_SIZE, (12, 8))

        self.assertIsNotNone(const.DPI)
        self.assertGreater(const.DPI, 0)
        self.assertEqual(const.DPI, 100)


class TestConstantsConsistency(unittest.TestCase):
    """Test consistency between related constants."""

    def test_timestep_simulation_time_consistency(self):
        """Test timestep and simulation time are consistent."""
        # Timestep should evenly divide simulation time
        n_steps = const.SIMULATION_TIME / const.DT
        self.assertAlmostEqual(n_steps, round(n_steps), places=10)
        self.assertEqual(int(n_steps), const.N_STEPS)

    def test_epsilon_consistency(self):
        """Test epsilon values are consistent."""
        # COLLISION_EPSILON should equal EPSILON
        self.assertEqual(const.COLLISION_EPSILON, const.EPSILON)

        # ENERGY_TOLERANCE should be larger than EPSILON
        self.assertGreater(const.ENERGY_TOLERANCE, const.EPSILON)

    def test_box_consistency(self):
        """Test box parameters are self-consistent."""
        # Width calculation
        self.assertAlmostEqual(
            const.BOX_WIDTH,
            const.BOX_MAX_X - const.BOX_MIN_X,
            places=10
        )

        # Height calculation
        self.assertAlmostEqual(
            const.BOX_HEIGHT,
            const.BOX_MAX_Y - const.BOX_MIN_Y,
            places=10
        )

    def test_physical_units_consistency(self):
        """Test physical constants have consistent units."""
        # All should be dimensionless in this simulation
        # Mass, charge are positive
        self.assertGreater(const.MASS, 0)
        self.assertGreater(const.CHARGE, 0)

        # Gravity is negative (downward)
        self.assertLess(const.GRAVITY, 0)

    def test_numerical_stability_requirements(self):
        """Test parameters meet numerical stability requirements."""
        # Timestep should be small enough for stability
        # For RK4, we need dt << characteristic time
        # Characteristic time ~ sqrt(L/g) ~ sqrt(100/10) ~ 3.16
        characteristic_time = np.sqrt(const.BOX_HEIGHT / abs(const.GRAVITY))
        self.assertLess(const.DT, characteristic_time / 100)

    def test_output_frequency_valid(self):
        """Test output frequency doesn't exceed total steps."""
        self.assertLessEqual(const.OUTPUT_FREQUENCY, const.N_STEPS)


class TestConstantsImmutability(unittest.TestCase):
    """Test that constants behave as constants (immutability)."""

    def test_constants_are_not_mutable_collections(self):
        """Test that array constants are numpy arrays (semi-immutable)."""
        self.assertIsInstance(const.INITIAL_STATES, np.ndarray)
        # Note: numpy arrays are mutable, but we trust users not to modify

    def test_numeric_constants_types(self):
        """Test numeric constants are immutable types."""
        immutable_types = (int, float, str, tuple)

        self.assertIsInstance(const.MASS, immutable_types)
        self.assertIsInstance(const.CHARGE, immutable_types)
        self.assertIsInstance(const.GRAVITY, immutable_types)
        self.assertIsInstance(const.DT, immutable_types)
        self.assertIsInstance(const.SIMULATION_TIME, immutable_types)
        self.assertIsInstance(const.OUTPUT_DIR, immutable_types)
        self.assertIsInstance(const.FIGURE_SIZE, immutable_types)


class TestConstantsCompleteness(unittest.TestCase):
    """Test that all required constants are defined."""

    def test_all_physical_constants_present(self):
        """Test all physical constants are defined."""
        required_physical = ['MASS', 'CHARGE', 'GRAVITY']
        for const_name in required_physical:
            self.assertTrue(hasattr(const, const_name),
                          f"Missing constant: {const_name}")

    def test_all_box_constants_present(self):
        """Test all box-related constants are defined."""
        required_box = [
            'BOX_MIN_X', 'BOX_MAX_X', 'BOX_MIN_Y', 'BOX_MAX_Y',
            'BOX_WIDTH', 'BOX_HEIGHT'
        ]
        for const_name in required_box:
            self.assertTrue(hasattr(const, const_name),
                          f"Missing constant: {const_name}")

    def test_all_numerical_constants_present(self):
        """Test all numerical constants are defined."""
        required_numerical = [
            'DT', 'SIMULATION_TIME', 'N_STEPS',
            'EPSILON', 'ENERGY_TOLERANCE',
            'COLLISION_EPSILON', 'MAX_COLLISION_ITERATIONS'
        ]
        for const_name in required_numerical:
            self.assertTrue(hasattr(const, const_name),
                          f"Missing constant: {const_name}")

    def test_all_output_constants_present(self):
        """Test all output-related constants are defined."""
        required_output = [
            'OUTPUT_DIR', 'OUTPUT_FILE', 'PLOT_DIR',
            'OUTPUT_FREQUENCY', 'FIGURE_SIZE', 'DPI'
        ]
        for const_name in required_output:
            self.assertTrue(hasattr(const, const_name),
                          f"Missing constant: {const_name}")

    def test_initial_conditions_present(self):
        """Test initial conditions are defined."""
        required_initial = ['INITIAL_STATES', 'N_PARTICLES']
        for const_name in required_initial:
            self.assertTrue(hasattr(const, const_name),
                          f"Missing constant: {const_name}")


class TestConstantsEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions for constants."""

    def test_zero_values_not_used_inappropriately(self):
        """Test that zero values aren't used where they shouldn't be."""
        # These should never be zero
        self.assertNotEqual(const.MASS, 0)
        self.assertNotEqual(const.CHARGE, 0)
        self.assertNotEqual(const.DT, 0)
        self.assertNotEqual(const.SIMULATION_TIME, 0)
        self.assertNotEqual(const.EPSILON, 0)
        self.assertNotEqual(const.BOX_WIDTH, 0)
        self.assertNotEqual(const.BOX_HEIGHT, 0)

    def test_reasonable_value_ranges(self):
        """Test constants are in reasonable ranges."""
        # Timestep should be reasonably small
        self.assertLess(const.DT, 1.0)
        self.assertGreater(const.DT, 1e-6)

        # Epsilon should be very small
        self.assertLess(const.EPSILON, 1e-8)
        self.assertGreater(const.EPSILON, 1e-15)

        # Box should be reasonably sized
        self.assertGreater(const.BOX_WIDTH, 1.0)
        self.assertLess(const.BOX_WIDTH, 1e6)

        # DPI should be reasonable
        self.assertGreaterEqual(const.DPI, 50)
        self.assertLessEqual(const.DPI, 300)

    def test_no_negative_where_positive_expected(self):
        """Test no negative values where positive expected."""
        self.assertGreater(const.MASS, 0)
        self.assertGreater(const.CHARGE, 0)
        self.assertGreater(const.DT, 0)
        self.assertGreater(const.SIMULATION_TIME, 0)
        self.assertGreater(const.BOX_WIDTH, 0)
        self.assertGreater(const.BOX_HEIGHT, 0)
        self.assertGreater(const.DPI, 0)

    def test_string_formats(self):
        """Test string constants have expected formats."""
        # Directory names shouldn't have spaces or special chars
        self.assertNotIn(' ', const.OUTPUT_DIR)
        self.assertNotIn(' ', const.PLOT_DIR)

        # File extension
        self.assertTrue(const.OUTPUT_FILE.endswith('.csv'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
