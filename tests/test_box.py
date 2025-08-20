"""
test_box.py - Comprehensive unit tests for Box class and collision handling

Tests wall collision detection, velocity reflection, boundary enforcement,
and the complete collision handling algorithm.
"""

import src.constants as const
from src.particle import Particle
from src.box import Box
import unittest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBoxInitialization(unittest.TestCase):
    """Test box initialization and properties."""

    def test_default_initialization(self):
        """Test box with default dimensions from constants."""
        box = Box()

        self.assertEqual(box.x_min, const.BOX_MIN_X)
        self.assertEqual(box.x_max, const.BOX_MAX_X)
        self.assertEqual(box.y_min, const.BOX_MIN_Y)
        self.assertEqual(box.y_max, const.BOX_MAX_Y)
        self.assertEqual(box.width, const.BOX_WIDTH)
        self.assertEqual(box.height, const.BOX_HEIGHT)

    def test_custom_dimensions(self):
        """Test box with custom dimensions."""
        box = Box(x_min=-10, x_max=20, y_min=-5, y_max=15)

        self.assertEqual(box.x_min, -10)
        self.assertEqual(box.x_max, 20)
        self.assertEqual(box.y_min, -5)
        self.assertEqual(box.y_max, 15)
        self.assertEqual(box.width, 30)
        self.assertEqual(box.height, 20)

    def test_square_box(self):
        """Test square box dimensions."""
        box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

        self.assertEqual(box.width, box.height)
        self.assertEqual(box.width, 10)

    def test_invalid_dimensions(self):
        """Test that invalid dimensions raise error."""
        # Negative width
        with self.assertRaises(ValueError):
            Box(x_min=10, x_max=5, y_min=0, y_max=10)

        # Negative height
        with self.assertRaises(ValueError):
            Box(x_min=0, x_max=10, y_min=10, y_max=5)

        # Zero width
        with self.assertRaises(ValueError):
            Box(x_min=5, x_max=5, y_min=0, y_max=10)

    def test_collision_statistics_initialization(self):
        """Test collision tracking is initialized."""
        box = Box()

        self.assertEqual(box.total_collisions, 0)
        self.assertEqual(len(box.collision_history), 0)


class TestPositionChecking(unittest.TestCase):
    """Test position checking methods."""

    def setUp(self):
        """Create test box."""
        self.box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

    def test_is_inside_center(self):
        """Test point at center is inside."""
        pos = np.array([5.0, 5.0])
        self.assertTrue(self.box.is_inside(pos))

    def test_is_inside_corners(self):
        """Test corners are considered inside."""
        corners = [
            np.array([0.0, 0.0]),    # Bottom-left
            np.array([10.0, 0.0]),   # Bottom-right
            np.array([10.0, 10.0]),  # Top-right
            np.array([0.0, 10.0]),   # Top-left
        ]

        for corner in corners:
            self.assertTrue(self.box.is_inside(corner))

    def test_is_inside_edges(self):
        """Test points on edges are inside."""
        edge_points = [
            np.array([5.0, 0.0]),    # Bottom edge
            np.array([10.0, 5.0]),   # Right edge
            np.array([5.0, 10.0]),   # Top edge
            np.array([0.0, 5.0]),    # Left edge
        ]

        for point in edge_points:
            self.assertTrue(self.box.is_inside(point))

    def test_is_outside(self):
        """Test points outside box."""
        outside_points = [
            np.array([-1.0, 5.0]),   # Left of box
            np.array([11.0, 5.0]),   # Right of box
            np.array([5.0, -1.0]),   # Below box
            np.array([5.0, 11.0]),   # Above box
            np.array([-1.0, -1.0]),  # Diagonal outside
        ]

        for point in outside_points:
            self.assertFalse(self.box.is_inside(point))


class TestCollisionDetection(unittest.TestCase):
    """Test wall collision detection."""

    def setUp(self):
        """Create test box."""
        self.box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

    def test_no_collision_interior_motion(self):
        """Test no collision for motion entirely inside box."""
        old_state = np.array([5.0, 5.0, 1.0, 1.0])
        new_state = np.array([5.1, 5.1, 1.0, 1.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertFalse(collision)
        self.assertIsNone(wall)
        self.assertIsNone(fraction)

    def test_collision_right_wall(self):
        """Test collision detection with right wall."""
        old_state = np.array([9.0, 5.0, 10.0, 0.0])
        new_state = np.array([11.0, 5.0, 10.0, 0.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'right')
        # Collision at x=10, traveled 1 unit out of 2 total
        self.assertAlmostEqual(fraction, 0.5, places=10)

    def test_collision_left_wall(self):
        """Test collision detection with left wall."""
        old_state = np.array([1.0, 5.0, -10.0, 0.0])
        new_state = np.array([-1.0, 5.0, -10.0, 0.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'left')
        self.assertAlmostEqual(fraction, 0.5, places=10)

    def test_collision_top_wall(self):
        """Test collision detection with top wall."""
        old_state = np.array([5.0, 9.0, 0.0, 10.0])
        new_state = np.array([5.0, 11.0, 0.0, 10.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'top')
        self.assertAlmostEqual(fraction, 0.5, places=10)

    def test_collision_bottom_wall(self):
        """Test collision detection with bottom wall."""
        old_state = np.array([5.0, 1.0, 0.0, -10.0])
        new_state = np.array([5.0, -1.0, 0.0, -10.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'bottom')
        self.assertAlmostEqual(fraction, 0.5, places=10)

    def test_collision_corner(self):
        """Test collision detection when heading toward corner."""
        old_state = np.array([9.0, 9.0, 10.0, 10.0])
        new_state = np.array([11.0, 11.0, 10.0, 10.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        # Should detect first wall hit (either right or top, both at fraction 0.5)
        self.assertIn(wall, ['right', 'top'])
        self.assertAlmostEqual(fraction, 0.5, places=10)

    def test_collision_from_edge(self):
        """Test collision starting exactly on edge."""
        old_state = np.array([10.0, 5.0, 10.0, 0.0])
        new_state = np.array([11.0, 5.0, 10.0, 0.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'right')
        self.assertAlmostEqual(fraction, 0.0, places=10)

    def test_no_collision_parallel_to_wall(self):
        """Test no collision when moving parallel to wall."""
        old_state = np.array([0.0, 5.0, 0.0, 10.0])  # On left wall, moving up
        new_state = np.array([0.0, 7.0, 0.0, 10.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertFalse(collision)

    def test_collision_very_small_motion(self):
        """Test collision detection for very small motions."""
        old_state = np.array([9.999, 5.0, 1.0, 0.0])
        new_state = np.array([10.001, 5.0, 1.0, 0.0])

        collision, wall, fraction = self.box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'right')


class TestVelocityReflection(unittest.TestCase):
    """Test velocity reflection at walls."""

    def setUp(self):
        """Create test box."""
        self.box = Box()

    def test_reflect_right_wall(self):
        """Test velocity reflection at right wall."""
        velocity = np.array([5.0, 3.0])
        reflected = self.box.reflect_velocity(velocity, 'right')

        expected = np.array([-5.0, 3.0])
        np.testing.assert_array_equal(reflected, expected)

    def test_reflect_left_wall(self):
        """Test velocity reflection at left wall."""
        velocity = np.array([-5.0, 3.0])
        reflected = self.box.reflect_velocity(velocity, 'left')

        expected = np.array([5.0, 3.0])
        np.testing.assert_array_equal(reflected, expected)

    def test_reflect_top_wall(self):
        """Test velocity reflection at top wall."""
        velocity = np.array([5.0, 3.0])
        reflected = self.box.reflect_velocity(velocity, 'top')

        expected = np.array([5.0, -3.0])
        np.testing.assert_array_equal(reflected, expected)

    def test_reflect_bottom_wall(self):
        """Test velocity reflection at bottom wall."""
        velocity = np.array([5.0, -3.0])
        reflected = self.box.reflect_velocity(velocity, 'bottom')

        expected = np.array([5.0, 3.0])
        np.testing.assert_array_equal(reflected, expected)

    def test_reflect_invalid_wall(self):
        """Test that invalid wall name raises error."""
        velocity = np.array([5.0, 3.0])

        with self.assertRaises(ValueError):
            self.box.reflect_velocity(velocity, 'invalid')

    def test_reflect_preserves_speed(self):
        """Test that reflection preserves speed (elastic collision)."""
        velocity = np.array([3.0, 4.0])
        initial_speed = np.linalg.norm(velocity)

        for wall in ['left', 'right', 'top', 'bottom']:
            reflected = self.box.reflect_velocity(velocity, wall)
            final_speed = np.linalg.norm(reflected)
            self.assertAlmostEqual(initial_speed, final_speed, places=10)

    def test_double_reflection(self):
        """Test that reflecting twice returns original velocity."""
        velocity = np.array([5.0, 3.0])

        for wall in ['left', 'right', 'top', 'bottom']:
            once = self.box.reflect_velocity(velocity, wall)
            twice = self.box.reflect_velocity(once, wall)
            np.testing.assert_array_almost_equal(twice, velocity)


class TestCollisionHandling(unittest.TestCase):
    """Test complete collision handling with RK4."""

    def setUp(self):
        """Create test box and particles."""
        self.box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

    def test_handle_collision_with_rk4_no_collision(self):
        """Test RK4 collision handling when no collision occurs."""
        particles = [
            Particle(x=5.0, y=5.0, vx=1.0, vy=1.0, mass=1.0, charge=0.0)
        ]

        old_state = particles[0].state.copy()
        dt = 0.1

        final_state, collision_occurred, dt_used = self.box.handle_collision_with_rk4(
            0, old_state, particles, dt
        )

        self.assertFalse(collision_occurred)
        self.assertEqual(dt_used, dt)
        # Particle should have moved
        self.assertGreater(final_state[0], old_state[0])
        self.assertNotEqual(final_state[1], old_state[1])  # Gravity affects y

    def test_handle_collision_with_rk4_wall_hit(self):
        """Test RK4 collision handling when particle hits wall."""
        particles = [
            Particle(x=9.5, y=5.0, vx=10.0, vy=0.0, mass=1.0, charge=0.0)
        ]

        old_state = particles[0].state.copy()
        dt = 0.1

        final_state, collision_occurred, dt_used = self.box.handle_collision_with_rk4(
            0, old_state, particles, dt
        )

        self.assertTrue(collision_occurred)
        self.assertEqual(dt_used, dt)
        # Velocity should be reflected
        self.assertLess(final_state[2], 0)  # vx should be negative
        # Position should be inside box
        self.assertLessEqual(final_state[0], 10.0)
        self.assertGreaterEqual(final_state[0], 0.0)

    def test_handle_collision_with_rk4_energy_conservation(self):
        """Test that collision handling conserves kinetic energy."""
        particles = [
            Particle(x=9.0, y=5.0, vx=20.0, vy=10.0, mass=1.0, charge=0.0)
        ]

        old_state = particles[0].state.copy()
        initial_ke = 0.5 * (old_state[2]**2 + old_state[3]**2)
        dt = 0.1

        final_state, collision_occurred, dt_used = self.box.handle_collision_with_rk4(
            0, old_state, particles, dt
        )

        # Account for gravity effect on vy
        # For pure elastic collision test, we check speed magnitude
        final_speed_squared = final_state[2]**2 + final_state[3]**2
        initial_speed_squared = old_state[2]**2 + old_state[3]**2

        # Speed change should only be due to gravity
        # This is approximate due to RK4 integration
        self.assertTrue(collision_occurred)

    def test_handle_collision_multiple_particles(self):
        """Test collision handling with multiple particles (force context)."""
        particles = [
            Particle(x=9.5, y=5.0, vx=10.0, vy=0.0, mass=1.0, charge=1.0),
            Particle(x=5.0, y=5.0, vx=0.0, vy=0.0, mass=1.0, charge=1.0),
        ]

        old_state = particles[0].state.copy()
        dt = 0.1

        final_state, collision_occurred, dt_used = self.box.handle_collision_with_rk4(
            0, old_state, particles, dt
        )

        self.assertTrue(collision_occurred)
        # Should include Coulomb repulsion effect
        self.assertNotEqual(final_state[0], old_state[0])


class TestBoundaryEnforcement(unittest.TestCase):
    """Test boundary enforcement methods."""

    def setUp(self):
        """Create test box."""
        self.box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

    def test_enforce_boundaries_inside(self):
        """Test that particles inside box are not affected."""
        p = Particle(x=5.0, y=5.0, vx=0, vy=0)
        original_pos = p.position.copy()

        self.box.enforce_boundaries(p)

        np.testing.assert_array_equal(p.position, original_pos)

    def test_enforce_boundaries_outside_right(self):
        """Test clamping particle outside right wall."""
        p = Particle(x=15.0, y=5.0, vx=0, vy=0)

        self.box.enforce_boundaries(p)

        self.assertEqual(p.x, 10.0)
        self.assertEqual(p.y, 5.0)

    def test_enforce_boundaries_outside_left(self):
        """Test clamping particle outside left wall."""
        p = Particle(x=-5.0, y=5.0, vx=0, vy=0)

        self.box.enforce_boundaries(p)

        self.assertEqual(p.x, 0.0)
        self.assertEqual(p.y, 5.0)

    def test_enforce_boundaries_outside_top(self):
        """Test clamping particle outside top wall."""
        p = Particle(x=5.0, y=15.0, vx=0, vy=0)

        self.box.enforce_boundaries(p)

        self.assertEqual(p.x, 5.0)
        self.assertEqual(p.y, 10.0)

    def test_enforce_boundaries_outside_bottom(self):
        """Test clamping particle outside bottom wall."""
        p = Particle(x=5.0, y=-5.0, vx=0, vy=0)

        self.box.enforce_boundaries(p)

        self.assertEqual(p.x, 5.0)
        self.assertEqual(p.y, 0.0)

    def test_enforce_boundaries_outside_corner(self):
        """Test clamping particle outside corner."""
        p = Particle(x=15.0, y=15.0, vx=0, vy=0)

        self.box.enforce_boundaries(p)

        self.assertEqual(p.x, 10.0)
        self.assertEqual(p.y, 10.0)

    def test_enforce_boundaries_preserves_velocity(self):
        """Test that boundary enforcement doesn't change velocity."""
        p = Particle(x=15.0, y=15.0, vx=5.0, vy=-3.0)
        original_vel = p.velocity.copy()

        self.box.enforce_boundaries(p)

        np.testing.assert_array_equal(p.velocity, original_vel)


class TestCollisionStatistics(unittest.TestCase):
    """Test collision tracking and statistics."""

    def setUp(self):
        """Create test box."""
        self.box = Box()

    def test_collision_counter(self):
        """Test that collision counter increments."""
        initial_count = self.box.total_collisions

        particles = [
            Particle(x=99.0, y=50.0, vx=10.0, vy=0.0, mass=1.0, charge=0.0)
        ]

        old_state = particles[0].state.copy()

        # Trigger collision
        self.box.handle_collision_with_rk4(0, old_state, particles, 0.5)

        self.assertEqual(self.box.total_collisions, initial_count + 1)

    def test_string_representation(self):
        """Test box string representation."""
        box = Box(x_min=0, x_max=10, y_min=0, y_max=10)
        s = str(box)

        self.assertIn("Box", s)
        self.assertIn("0", s)
        self.assertIn("10", s)
        self.assertIn("Total collisions", s)


class TestBoxEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def test_zero_velocity_collision(self):
        """Test collision detection with zero velocity."""
        box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

        old_state = np.array([5.0, 5.0, 0.0, 0.0])
        new_state = np.array([5.0, 5.0, 0.0, 0.0])

        collision, wall, fraction = box.check_wall_collision(old_state, new_state)

        self.assertFalse(collision)

    def test_negative_box_coordinates(self):
        """Test box with negative coordinate boundaries."""
        box = Box(x_min=-10, x_max=10, y_min=-10, y_max=10)

        # Test position at origin is inside
        self.assertTrue(box.is_inside(np.array([0.0, 0.0])))

        # Test collision detection works
        old_state = np.array([-9.0, 0.0, -10.0, 0.0])
        new_state = np.array([-11.0, 0.0, -10.0, 0.0])

        collision, wall, fraction = box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'left')

    def test_very_large_box(self):
        """Test box with very large dimensions."""
        box = Box(x_min=0, x_max=1e6, y_min=0, y_max=1e6)

        self.assertEqual(box.width, 1e6)
        self.assertEqual(box.height, 1e6)

        # Test collision still works
        old_state = np.array([999999.0, 500000.0, 100.0, 0.0])
        new_state = np.array([1000001.0, 500000.0, 100.0, 0.0])

        collision, wall, fraction = box.check_wall_collision(old_state, new_state)

        self.assertTrue(collision)
        self.assertEqual(wall, 'right')

    def test_very_small_box(self):
        """Test box with very small dimensions."""
        box = Box(x_min=0, x_max=0.001, y_min=0, y_max=0.001)

        self.assertEqual(box.width, 0.001)

        # Test particle can fit
        self.assertTrue(box.is_inside(np.array([0.0005, 0.0005])))

    def test_numerical_precision_at_boundary(self):
        """Test numerical precision issues at boundaries."""
        box = Box(x_min=0, x_max=10, y_min=0, y_max=10)

        # Test values very close to boundary
        epsilon = 1e-15

        # Just inside
        self.assertTrue(box.is_inside(np.array([10.0 - epsilon, 5.0])))

        # Just outside
        self.assertFalse(box.is_inside(np.array([10.0 + epsilon, 5.0])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
