"""
integrator.py - Runge-Kutta 4th order (RK4) numerical integrator

This module implements the RK4 integration method for solving the
particle motion differential equations. The second-order ODE is
converted to a first-order system using state vectors.

The state vector is [x, y, vx, vy] and the derivative is [vx, vy, ax, ay].

Author: Simulation Team
Date: 2024
"""

import numpy as np
from typing import Callable, List
from . import constants as const
from .particle import Particle
from .forces import calculate_system_accelerations


def state_derivative(particles: List[Particle]) -> List[np.ndarray]:
    """
    Calculate the derivative of the state vector for all particles.

    For state vector s = [x, y, vx, vy], the derivative is:
    ds/dt = [vx, vy, ax, ay]

    where accelerations are calculated from forces.
    """
    # Calculate accelerations for all particles
    accelerations = calculate_system_accelerations(particles)

    derivatives = []
    for particle, acceleration in zip(particles, accelerations):
        # Extract velocity from state
        velocity = particle.velocity

        # Construct derivative: [vx, vy, ax, ay]
        derivative = np.zeros(4)
        derivative[0:2] = velocity  # dx/dt = vx, dy/dt = vy
        derivative[2:4] = acceleration  # dvx/dt = ax, dvy/dt = ay

        derivatives.append(derivative)

    return derivatives


def rk4_step_single(particle: Particle,
                    all_particles: List[Particle],
                    particle_index: int,
                    dt: float) -> np.ndarray:
    """
    Perform one RK4 integration step for a single particle.

    This function calculates the state increment for one particle
    considering forces from all other particles. This is needed for
    collision handling where we need to recalculate the trajectory
    of a single particle after a collision.

    Args:
        particle: The particle to integrate
        all_particles: List of all particles (for force calculations)
        particle_index: Index of the particle in the list
        dt: Timestep size

    Returns:
        np.ndarray: State increment vector
    """
    # Handle zero timestep
    if abs(dt) < 1e-15:
        return np.zeros(4)

    # Store original states of all particles
    original_states = [p.state.copy() for p in all_particles]

    try:
        # Calculate k1
        # k1 = dt * f(s_n)
        derivatives_k1 = state_derivative(all_particles)
        k1 = dt * derivatives_k1[particle_index]

        # Calculate k2
        # Move all particles to s_n + k1/2 for consistent force calculation
        for i, p in enumerate(all_particles):
            if i == particle_index:
                p.update_state(original_states[i] + 0.5 * k1)
            else:
                # Other particles move with their own k1
                p.update_state(original_states[i] + 0.5 * dt * derivatives_k1[i])

        derivatives_k2 = state_derivative(all_particles)
        k2 = dt * derivatives_k2[particle_index]

        # Calculate k3
        # Move all particles to s_n + k2/2
        for i, p in enumerate(all_particles):
            if i == particle_index:
                p.update_state(original_states[i] + 0.5 * k2)
            else:
                # Other particles move with their own k2
                p.update_state(original_states[i] + 0.5 * dt * derivatives_k2[i])

        derivatives_k3 = state_derivative(all_particles)
        k3 = dt * derivatives_k3[particle_index]

        # Calculate k4
        # Move all particles to s_n + k3
        for i, p in enumerate(all_particles):
            if i == particle_index:
                p.update_state(original_states[i] + k3)
            else:
                # Other particles move with their own k3
                p.update_state(original_states[i] + dt * derivatives_k3[i])

        derivatives_k4 = state_derivative(all_particles)
        k4 = dt * derivatives_k4[particle_index]

        # Combine RK4 coefficients
        # increment = (k1 + 2*k2 + 2*k3 + k4) / 6
        state_increment = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    finally:
        # Always restore all particles to original states
        for i, p in enumerate(all_particles):
            p.update_state(original_states[i])

    return state_increment


def rk4_step_system(particles: List[Particle], dt: float) -> List[np.ndarray]:
    """
    Perform one RK4 integration step for the entire particle system.

    This implements the batch update approach where all particle forces
    are calculated before any states are updated. This is the standard
    approach for coupled systems of ODEs.

    Args:
        particles: List of all particles
        dt: Timestep size

    Returns:
        List[np.ndarray]: List of state increments for all particles
    """
    # Handle zero timestep
    if abs(dt) < 1e-15:
        return [np.zeros(4) for _ in particles]

    n_particles = len(particles)

    # Store original states
    original_states = [p.state.copy() for p in particles]

    # Initialize storage for RK4 coefficients
    k1_list = []
    k2_list = []
    k3_list = []
    k4_list = []

    try:
        # Calculate k1 for all particles
        # k1 = dt * f(s_n)
        derivatives = state_derivative(particles)
        for derivative in derivatives:
            k1_list.append(dt * derivative)

        # Calculate k2 for all particles
        # Move all particles to s_n + k1/2
        for i, particle in enumerate(particles):
            particle.update_state(original_states[i] + 0.5 * k1_list[i])

        # Calculate derivatives at midpoint
        derivatives = state_derivative(particles)
        for derivative in derivatives:
            k2_list.append(dt * derivative)

        # Calculate k3 for all particles
        # Move all particles to s_n + k2/2
        for i, particle in enumerate(particles):
            particle.update_state(original_states[i] + 0.5 * k2_list[i])

        derivatives = state_derivative(particles)
        for derivative in derivatives:
            k3_list.append(dt * derivative)

        # Calculate k4 for all particles
        # Move all particles to s_n + k3
        for i, particle in enumerate(particles):
            particle.update_state(original_states[i] + k3_list[i])

        derivatives = state_derivative(particles)
        for derivative in derivatives:
            k4_list.append(dt * derivative)

        # Combine RK4 coefficients for each particle
        state_increments = []
        for i in range(n_particles):
            increment = (k1_list[i] + 2 * k2_list[i] + 2 * k3_list[i] + k4_list[i]) / 6.0
            state_increments.append(increment)

    finally:
        # Always restore original states
        for i, particle in enumerate(particles):
            particle.update_state(original_states[i])

    return state_increments


class RK4Integrator:
    """
    Runge-Kutta 4th order integrator for particle system.

    This class encapsulates the RK4 integration method and provides
    a clean interface for time-stepping the particle system.
    """

    def __init__(self, dt: float = const.DT):
        """
        Initialize the RK4 integrator.

        Args:
            dt: Timestep size
        """
        self.dt = dt
        self.step_count = 0
        self.total_time = 0.0

    def step(self, particles: List[Particle]) -> List[np.ndarray]:
        """
        Advance the particle system by one timestep.

        Args:
            particles: List of particles to integrate

        Returns:
            List[np.ndarray]: State increments for all particles
        """
        # Calculate state increments using RK4
        increments = rk4_step_system(particles, self.dt)

        # Update statistics
        self.step_count += 1
        self.total_time += self.dt

        return increments

    def integrate_to_time(self,
                          particles: List[Particle],
                          target_time: float,
                          callback: Callable = None) -> List[Particle]:
        """
        Integrate the system to a target time.

        Args:
            particles: List of particles
            target_time: Time to integrate to
            callback: Optional function called after each step

        Returns:
            List[Particle]: Updated particle list
        """
        while abs(self.total_time - target_time) > const.EPSILON:
            # Calculate remaining time
            remaining_time = target_time - self.total_time
            dt_step = min(abs(self.dt), abs(remaining_time)) * np.sign(remaining_time)

            # Perform integration step
            increments = rk4_step_system(particles, dt_step)

            # Update particle states
            for particle, increment in zip(particles, increments):
                particle.update_state(particle.state + increment)

            # Update time
            self.total_time += dt_step
            self.step_count += 1

            # Call callback if provided
            if callback is not None:
                callback(self.total_time, particles)

        return particles

    def reset(self):
        """Reset the integrator statistics."""
        self.step_count = 0
        self.total_time = 0.0
