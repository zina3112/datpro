"""
box.py - Simulation box with EXACT interpolation-based wall collision handling

This module implements the EXACT collision detection and handling method
as specified in the project requirements. It uses linear interpolation
to find the precise collision time and splits the RK4 integration accordingly.

CRITICAL: This implementation follows the specification EXACTLY as required.
"""

import numpy as np
from typing import Tuple, Optional, List
from . import constants as const
from .particle import Particle


class Box:
    """
    Represents the 2D simulation box with perfectly elastic wall collisions.

    This implementation uses the EXACT method specified in the project:
    1. Calculate full RK4 step
    2. Check if particle would exit box
    3. Use LINEAR INTERPOLATION to find collision time
    4. Split timestep at collision point
    5. Apply reflection and continue
    """

    def __init__(self,
                 x_min: float = const.BOX_MIN_X,
                 x_max: float = const.BOX_MAX_X,
                 y_min: float = const.BOX_MIN_Y,
                 y_max: float = const.BOX_MAX_Y):
        """Initialize the simulation box with boundaries."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.width = x_max - x_min
        self.height = y_max - y_min

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

        # Statistics tracking
        self.total_collisions = 0
        self.collision_history = []

    def is_inside(self, position: np.ndarray) -> bool:
        """
        Check if a position is inside the box boundaries.

        Args:
            position: [x, y] position vector

        Returns:
            bool: True if position is inside box (including boundaries)
        """
        x, y = position
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def handle_wall_collision_exact(self,
                                    particle: Particle,
                                    all_particles: List[Particle],
                                    particle_index: int,
                                    dt: float) -> np.ndarray:
        """
        Handle wall collisions using the EXACT interpolation method from the specification.

        This is the MAIN METHOD that implements the required algorithm:

        1. First calculate the full RK4 step for dt
        2. Check if the resulting position would be outside the box
        3. If yes, use LINEAR INTERPOLATION to find the exact collision time
        4. Re-do RK4 for only the pre-collision fraction of dt
        5. Reflect the velocity at the collision point
        6. Do another RK4 for the remaining time with reflected velocity

        Args:
            particle: The particle to update
            all_particles: All particles (needed for force calculations)
            particle_index: Index of this particle in the list
            dt: The timestep

        Returns:
            np.ndarray: The final state after handling any collisions
        """
        from .integrator import rk4_step_single

        # Store the original state before any calculations
        original_state = particle.state.copy()

        # STEP 1: Calculate the full RK4 step as if there were no walls
        # This gives us the "tentative" new state
        full_step_increment = rk4_step_single(
            particle,
            all_particles,
            particle_index,
            dt
        )

        # Calculate what the new state would be after the full timestep
        tentative_new_state = original_state + full_step_increment

        # STEP 2: Check if this tentative position is outside the box
        tentative_position = tentative_new_state[0:2]

        if self.is_inside(tentative_position):
            # No collision - return the full step result
            return tentative_new_state

        # STEP 3: A collision will occur - find WHEN using LINEAR INTERPOLATION
        # We need to find the fraction of dt at which the particle hits the wall

        # Extract positions for interpolation
        x0, y0 = original_state[0:2]  # Starting position
        x1, y1 = tentative_new_state[0:2]  # Would-be ending position

        # Calculate the collision time fraction for each wall
        # We find the EARLIEST collision (smallest positive fraction)
        collision_fraction = 1.0  # Initialize to full timestep
        wall_hit = None

        # Check collision with LEFT wall (x = x_min)
        if x1 < self.x_min and x0 >= self.x_min:
            # Particle crosses left wall
            # Linear interpolation: find t such that x(t) = x_min
            # x(t) = x0 + t*(x1-x0) = x_min
            # Therefore: t = (x_min - x0)/(x1 - x0)
            if abs(x1 - x0) > const.EPSILON:  # Avoid division by zero
                t_left = (self.x_min - x0) / (x1 - x0)
                if 0 <= t_left < collision_fraction:
                    collision_fraction = t_left
                    wall_hit = 'left'

        # Check collision with RIGHT wall (x = x_max)
        elif x1 > self.x_max and x0 <= self.x_max:
            # Particle crosses right wall
            # Linear interpolation: t = (x_max - x0)/(x1 - x0)
            if abs(x1 - x0) > const.EPSILON:
                t_right = (self.x_max - x0) / (x1 - x0)
                if 0 <= t_right < collision_fraction:
                    collision_fraction = t_right
                    wall_hit = 'right'

        # Check collision with BOTTOM wall (y = y_min)
        if y1 < self.y_min and y0 >= self.y_min:
            # Particle crosses bottom wall
            # Linear interpolation: t = (y_min - y0)/(y1 - y0)
            if abs(y1 - y0) > const.EPSILON:
                t_bottom = (self.y_min - y0) / (y1 - y0)
                if 0 <= t_bottom < collision_fraction:
                    collision_fraction = t_bottom
                    wall_hit = 'bottom'

        # Check collision with TOP wall (y = y_max)
        elif y1 > self.y_max and y0 <= self.y_max:
            # Particle crosses top wall
            # Linear interpolation: t = (y_max - y0)/(y1 - y0)
            if abs(y1 - y0) > const.EPSILON:
                t_top = (self.y_max - y0) / (y1 - y0)
                if 0 <= t_top < collision_fraction:
                    collision_fraction = t_top
                    wall_hit = 'top'

        # STEP 4: Execute RK4 for only the pre-collision fraction of dt
        # This brings the particle exactly to the wall
        dt_to_collision = collision_fraction * dt

        if dt_to_collision > const.EPSILON:
            # Calculate RK4 step up to the collision point
            step_to_collision = rk4_step_single(
                particle,
                all_particles,
                particle_index,
                dt_to_collision
            )
            state_at_collision = original_state + step_to_collision
        else:
            # Collision happens immediately
            state_at_collision = original_state.copy()

        # STEP 5: Reflect the velocity component perpendicular to the wall
        # This implements perfect elastic reflection
        reflected_state = state_at_collision.copy()

        if wall_hit == 'left' or wall_hit == 'right':
            # Reflect x-velocity (reverse sign)
            reflected_state[2] = -reflected_state[2]
            # Record collision for statistics
            self.total_collisions += 1
            particle.collision_count += 1

        elif wall_hit == 'bottom' or wall_hit == 'top':
            # Reflect y-velocity (reverse sign)
            reflected_state[3] = -reflected_state[3]
            # Record collision for statistics
            self.total_collisions += 1
            particle.collision_count += 1

        # STEP 6: Execute RK4 for the remaining time after collision
        # Use the reflected velocity for this portion
        dt_after_collision = dt - dt_to_collision

        if dt_after_collision > const.EPSILON:
            # Temporarily update particle state to reflected state
            # This is needed for force calculations in RK4
            particle.update_state(reflected_state)

            # Calculate RK4 step for remaining time
            step_after_collision = rk4_step_single(
                particle,
                all_particles,
                particle_index,
                dt_after_collision
            )

            # Restore original state (important for batch updates)
            particle.update_state(original_state)

            # Final state is reflected state plus the post-collision step
            final_state = reflected_state + step_after_collision
        else:
            # No time remaining after collision
            final_state = reflected_state

        # SAFETY CHECK: Ensure particle is inside box after all calculations
        # This handles numerical errors and corner collisions
        if not self.is_inside(final_state[0:2]):
            # Clamp position to box boundaries
            final_state[0] = np.clip(final_state[0], self.x_min, self.x_max)
            final_state[1] = np.clip(final_state[1], self.y_min, self.y_max)

        # Check for secondary collision (particle might hit another wall)
        # This is important for corners and high-speed particles
        if not self.is_inside(final_state[0:2]) or self._would_exit_in_next_step(final_state, dt * 0.1):
            # Recursively handle secondary collision with remaining time
            # This ensures proper handling of corner bounces
            if dt_after_collision > const.EPSILON and self.total_collisions < const.MAX_COLLISION_ITERATIONS:
                # Update particle temporarily for recursive call
                particle.update_state(reflected_state)
                final_state = self.handle_wall_collision_exact(
                    particle,
                    all_particles,
                    particle_index,
                    dt_after_collision
                )
                # Restore original state
                particle.update_state(original_state)

        return final_state

    def _would_exit_in_next_step(self, state: np.ndarray, dt: float) -> bool:
        """
        Helper method to check if particle would exit box in next step.

        This is used to detect potential corner collisions.

        Args:
            state: Current state vector [x, y, vx, vy]
            dt: Timestep to check

        Returns:
            bool: True if particle would exit box
        """
        # Simple linear projection
        next_x = state[0] + state[2] * dt
        next_y = state[1] + state[3] * dt

        return not self.is_inside(np.array([next_x, next_y]))

    def check_and_handle_collisions_simple(self, particle: Particle, dt: float) -> bool:
        """
        Simple collision handling for post-step correction.

        This is a backup method that can be used after the main integration
        to ensure particles stay in bounds. It's less accurate than the
        interpolation method but provides a safety net.

        Args:
            particle: Particle to check
            dt: Timestep (for recording purposes)

        Returns:
            bool: True if a collision was handled
        """
        collision_occurred = False

        # Check x boundaries
        if particle.x < self.x_min:
            particle.state[0] = 2 * self.x_min - particle.x  # Reflect position
            particle.state[2] = -particle.state[2]  # Reverse x velocity
            collision_occurred = True

        elif particle.x > self.x_max:
            particle.state[0] = 2 * self.x_max - particle.x  # Reflect position
            particle.state[2] = -particle.state[2]  # Reverse x velocity
            collision_occurred = True

        # Check y boundaries
        if particle.y < self.y_min:
            particle.state[1] = 2 * self.y_min - particle.y  # Reflect position
            particle.state[3] = -particle.state[3]  # Reverse y velocity
            collision_occurred = True

        elif particle.y > self.y_max:
            particle.state[1] = 2 * self.y_max - particle.y  # Reflect position
            particle.state[3] = -particle.state[3]  # Reverse y velocity
            collision_occurred = True

        if collision_occurred:
            self.total_collisions += 1
            particle.collision_count += 1
            particle.last_collision_time = dt

        return collision_occurred

    def enforce_boundaries(self, particle: Particle):
        """
        Ensure particle stays within box boundaries.

        This is a final safety check to handle any numerical errors.

        Args:
            particle: Particle to constrain
        """
        particle.state[0] = np.clip(particle.state[0], self.x_min, self.x_max)
        particle.state[1] = np.clip(particle.state[1], self.y_min, self.y_max)

    def __str__(self) -> str:
        """String representation of the box."""
        return (f"Box: [{self.x_min}, {self.x_max}] × [{self.y_min}, {self.y_max}], "
                f"Total collisions: {self.total_collisions}")
