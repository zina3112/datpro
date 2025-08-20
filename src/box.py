"""
box.py - Simulation box with wall collision handling

This module defines the Box class which represents the 2D simulation domain
with perfectly elastic wall reflections.
"""

import numpy as np
from typing import Tuple, Optional, List
from . import constants as const
from .particle import Particle


class Box:
    """
    Represents the 2D simulation box with elastic wall collisions.
    """

    def __init__(self,
                 x_min: float = const.BOX_MIN_X,
                 x_max: float = const.BOX_MAX_X,
                 y_min: float = const.BOX_MIN_Y,
                 y_max: float = const.BOX_MAX_Y):
        """Initialize the simulation box."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.width = x_max - x_min
        self.height = y_max - y_min

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Box dimensions must be positive")

        self.total_collisions = 0
        self.collision_history = []

    def is_inside(self, position: np.ndarray) -> bool:
        """Check if a position is inside the box."""
        x, y = position
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def check_and_handle_collisions(self, particle: Particle, dt: float) -> bool:
        """
        Check and handle wall collisions for a particle after it has moved.

        This simplified approach directly reflects velocities if particle is outside box.
        Returns True if collision occurred.
        """
        collision_occurred = False

        # Check x boundaries
        if particle.x < self.x_min:
            particle.state[0] = self.x_min + (self.x_min - particle.x)  # Reflect position
            particle.state[2] = -particle.state[2]  # Reverse x velocity
            collision_occurred = True
            particle.collision_count += 1
            self.total_collisions += 1
        elif particle.x > self.x_max:
            particle.state[0] = self.x_max - (particle.x - self.x_max)  # Reflect position
            particle.state[2] = -particle.state[2]  # Reverse x velocity
            collision_occurred = True
            particle.collision_count += 1
            self.total_collisions += 1

        # Check y boundaries
        if particle.y < self.y_min:
            particle.state[1] = self.y_min + (self.y_min - particle.y)  # Reflect position
            particle.state[3] = -particle.state[3]  # Reverse y velocity
            collision_occurred = True
            particle.collision_count += 1
            self.total_collisions += 1
        elif particle.y > self.y_max:
            particle.state[1] = self.y_max - (particle.y - self.y_max)  # Reflect position
            particle.state[3] = -particle.state[3]  # Reverse y velocity
            collision_occurred = True
            particle.collision_count += 1
            self.total_collisions += 1

        return collision_occurred

    def check_wall_collision(self,
                             old_state: np.ndarray,
                             new_state: np.ndarray) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Check if particle trajectory crosses a wall.

        Returns:
            Tuple of (collision_occurred, wall_hit, collision_fraction)
        """
        x_old, y_old = old_state[0:2]
        x_new, y_new = new_state[0:2]

        # Check if new position is outside box
        if self.is_inside(new_state[0:2]):
            return False, None, None

        # Find earliest collision time
        collision_time = 1.0  # Full timestep
        wall_hit = None

        # Check collision with left wall
        if x_new < self.x_min and x_old >= self.x_min:
            if abs(x_new - x_old) > const.EPSILON:
                t = (self.x_min - x_old) / (x_new - x_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'left'

        # Check collision with right wall
        if x_new > self.x_max and x_old <= self.x_max:
            if abs(x_new - x_old) > const.EPSILON:
                t = (self.x_max - x_old) / (x_new - x_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'right'

        # Check collision with bottom wall
        if y_new < self.y_min and y_old >= self.y_min:
            if abs(y_new - y_old) > const.EPSILON:
                t = (self.y_min - y_old) / (y_new - y_old)
                if 0 <= t < collision_time:
                    collision_time = t
                    wall_hit = 'bottom'

        # Check collision with top wall
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
        """Reflect velocity vector based on which wall was hit."""
        vx, vy = velocity

        if wall == 'left' or wall == 'right':
            vx = -vx
        elif wall == 'bottom' or wall == 'top':
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
        Handle wall collision with simplified approach.
        """
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
            return new_state, False, dt

        # Handle collision with simplified approach
        self.total_collisions += 1
        all_particles[particle_index].collision_count += 1
        all_particles[particle_index].last_collision_time = collision_fraction * dt

        # Move to collision point
        dt_to_collision = collision_fraction * dt
        if dt_to_collision > const.EPSILON:
            increment_to_collision = rk4_step_single(
                all_particles[particle_index],
                all_particles,
                particle_index,
                dt_to_collision
            )
            state_at_collision = old_state + increment_to_collision
        else:
            state_at_collision = old_state.copy()

        # Reflect velocity
        velocity_at_collision = state_at_collision[2:4]
        reflected_velocity = self.reflect_velocity(velocity_at_collision, wall)

        state_after_reflection = state_at_collision.copy()
        state_after_reflection[2:4] = reflected_velocity

        # Continue with remaining time
        original_particle_state = all_particles[particle_index].state.copy()
        all_particles[particle_index].update_state(state_after_reflection)

        remaining_dt = dt - dt_to_collision
        if remaining_dt > const.EPSILON:
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

        # Ensure particle is inside box
        if not self.is_inside(final_state[0:2]):
            final_state[0] = np.clip(final_state[0], self.x_min, self.x_max)
            final_state[1] = np.clip(final_state[1], self.y_min, self.y_max)

        return final_state, True, dt

    def enforce_boundaries(self, particle: Particle):
        """Ensure particle stays within box boundaries."""
        particle.state[0] = np.clip(particle.state[0], self.x_min, self.x_max)
        particle.state[1] = np.clip(particle.state[1], self.y_min, self.y_max)

    def __str__(self) -> str:
        """String representation of the box."""
        return (f"Box: [{self.x_min}, {self.x_max}] × [{self.y_min}, {self.y_max}], "
                f"Total collisions: {self.total_collisions}")
