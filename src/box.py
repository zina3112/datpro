"""
box.py - Simulation box with wall collision handling

This module defines the Box class which represents the 2D simulation domain
with perfectly elastic wall reflections. It handles collision detection and
velocity reflection for particles hitting the box boundaries.

The collision handling uses linear interpolation to find the exact collision
time and splits the timestep accordingly.

Author: Simulation Team
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional, List
from . import constants as const
from .particle import Particle


class Box:
    """
    Represents the 2D simulation box with elastic wall collisions.

    The box has boundaries at [x_min, x_max] × [y_min, y_max].
    When particles hit the walls, they undergo perfect elastic reflection
    where the velocity component perpendicular to the wall is reversed.

    Attributes:
        x_min, x_max: Horizontal boundaries
        y_min, y_max: Vertical boundaries
        width, height: Box dimensions
    """

    def __init__(self,
                 x_min: float = const.BOX_MIN_X,
                 x_max: float = const.BOX_MAX_X,
                 y_min: float = const.BOX_MIN_Y,
                 y_max: float = const.BOX_MAX_Y):
        """
        Initialize the simulation box.

        Args:
            x_min: Left boundary
            x_max: Right boundary
            y_min: Bottom boundary
            y_max: Top boundary
        """
        # Store box boundaries
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        # Calculate box dimensions
        self.width = x_max - x_min
        self.height = y_max - y_min

        # Validate box dimensions
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

        # Statistics for debugging
        self.total_collisions = 0
        self.collision_history = []

    def is_inside(self, position: np.ndarray) -> bool:
        """
        Check if a position is inside the box.

        Args:
            position: Position vector [x, y]

        Returns:
            bool: True if position is inside box boundaries
        """
        x, y = position
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def check_wall_collision(self,
                             old_state: np.ndarray,
                             new_state: np.ndarray) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if particle trajectory crosses a wall.

        This function checks if the particle path from old_state to new_state
        crosses any box boundary. If so, it returns which wall was hit first
        and the fraction of the timestep before collision.

        Args:
            old_state: Initial state vector [x, y, vx, vy]
            new_state: Final state vector [x, y, vx, vy]

        Returns:
            Tuple of (collision_occurred, wall_hit, collision_fraction)
            - collision_occurred: True if collision detected
            - wall_hit: 'left', 'right', 'bottom', or 'top'
            - collision_fraction: Fraction of timestep before collision (0 to 1)
        """
        x_old, y_old = old_state[0:2]
        x_new, y_new = new_state[0:2]

        # Check if new position is outside box
        if self.is_inside(new_state[0:2]):
            return False, None, None

        # Find earliest collision time
        collision_time = 1.0  # Full timestep
        wall_hit = None

        # Check collision with left wall (x = x_min)
        if x_new < self.x_min and x_old >= self.x_min:
            # Linear interpolation to find collision time
            # x(t) = x_old + t * (x_new - x_old) = x_min
            # Solve for t: t = (x_min - x_old) / (x_new - x_old)
            if abs(x_new - x_old) > const.EPSILON:
                t = (self.x_min - x_old) / (x_new - x_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'left'

        # Check collision with right wall (x = x_max)
        if x_new > self.x_max and x_old <= self.x_max:
            if abs(x_new - x_old) > const.EPSILON:
                t = (self.x_max - x_old) / (x_new - x_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'right'

        # Check collision with bottom wall (y = y_min)
        if y_new < self.y_min and y_old >= self.y_min:
            if abs(y_new - y_old) > const.EPSILON:
                t = (self.y_min - y_old) / (y_new - y_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'bottom'

        # Check collision with top wall (y = y_max)
        if y_new > self.y_max and y_old <= self.y_max:
            if abs(y_new - y_old) > const.EPSILON:
                t = (self.y_max - y_old) / (y_new - y_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'top'

        if wall_hit is not None:
            return True, wall_hit, collision_time

        return False, None, None

    def reflect_velocity(self, velocity: np.ndarray, wall: str) -> np.ndarray:
        """
        Reflect velocity vector based on which wall was hit.

        For elastic collision:
        - Horizontal walls (top/bottom): reverse y-velocity
        - Vertical walls (left/right): reverse x-velocity

        Args:
            velocity: Velocity vector [vx, vy]
            wall: Which wall was hit ('left', 'right', 'top', 'bottom')

        Returns:
            np.ndarray: Reflected velocity vector
        """
        vx, vy = velocity

        if wall == 'left' or wall == 'right':
            # Reverse x-component for vertical walls
            vx = -vx
        elif wall == 'bottom' or wall == 'top':
            # Reverse y-component for horizontal walls
            vy = -vy
        else:
            raise ValueError(f"Unknown wall: {wall}")

        return np.array([vx, vy])

    def handle_collision_with_rk4(self,
                                  particle_index: int,
                                  old_state: np.ndarray,
                                  all_particles: List[Particle],
                                  dt: float) -> Tuple[np.ndarray, bool, float]:
        """
        Handle wall collision with proper RK4 recalculation.

        This method properly handles wall collisions by:
        1. Computing RK4 up to the collision point
        2. Reflecting the velocity
        3. Computing RK4 for the remaining time

        It also updates collision statistics for the particle.
        """
        # Import here to avoid circular dependency
        from .integrator import rk4_step_single

        # Calculate full RK4 step
        full_increment = rk4_step_single(
            all_particles[particle_index],
            all_particles,
            particle_index,
            dt
        )

        new_state = old_state + full_increment

        # Check for collision
        collision, wall, collision_fraction = self.check_wall_collision(old_state, new_state)

        if not collision:
            # No collision, return new state
            return new_state, False, dt

        # Collision detected - handle it properly
        self.total_collisions += 1

        # Update the particle's collision count and time
        all_particles[particle_index].collision_count += 1
        all_particles[particle_index].last_collision_time = collision_fraction * dt

        # Step 1: Recalculate RK4 up to collision time
        dt_to_collision = collision_fraction * dt
        if dt_to_collision > const.EPSILON:  # Only if there's time before collision
            increment_to_collision = rk4_step_single(
                all_particles[particle_index],
                all_particles,
                particle_index,
                dt_to_collision
            )
            state_at_collision = old_state + increment_to_collision
        else:
            state_at_collision = old_state.copy()

        # Step 2: Reflect velocity at collision
        velocity_at_collision = state_at_collision[2:4]
        reflected_velocity = self.reflect_velocity(velocity_at_collision, wall)

        # Create state after reflection
        state_after_reflection = state_at_collision.copy()
        state_after_reflection[2:4] = reflected_velocity

        # Step 3: Update particle state for force calculation
        original_particle_state = all_particles[particle_index].state.copy()
        all_particles[particle_index].update_state(state_after_reflection)

        # Step 4: Calculate RK4 for remaining time with NEW forces
        remaining_dt = dt - dt_to_collision
        if remaining_dt > const.EPSILON:  # Only if there's remaining time
            remaining_increment = rk4_step_single(
                all_particles[particle_index],
                all_particles,
                particle_index,
                remaining_dt
            )
            final_state = state_after_reflection + remaining_increment
        else:
            final_state = state_after_reflection

        # Restore original particle state
        all_particles[particle_index].update_state(original_particle_state)

        # Safety check: ensure particle is inside box
        if not self.is_inside(final_state[0:2]):
            # Clamp position to box boundaries as safety measure
            final_state[0] = np.clip(final_state[0], self.x_min, self.x_max)
            final_state[1] = np.clip(final_state[1], self.y_min, self.y_max)

        return final_state, True, dt

    def enforce_boundaries(self, particle: Particle):
        """
        Ensure particle stays within box boundaries (safety check).

        This is a fail-safe to handle any numerical errors that might
        place a particle outside the box.

        Args:
            particle: Particle to check and correct if needed
        """
        # Clamp position to box boundaries
        particle.state[0] = np.clip(particle.state[0], self.x_min, self.x_max)
        particle.state[1] = np.clip(particle.state[1], self.y_min, self.y_max)

    def __str__(self) -> str:
        """String representation of the box."""
        return (f"Box: [{self.x_min}, {self.x_max}] × [{self.y_min}, {self.y_max}], "
                f"Total collisions: {self.total_collisions}")
