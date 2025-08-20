"""
forces.py - Force calculation module for charged particle simulation

This module handles all force calculations including gravitational forces
and electrostatic (Coulomb) repulsion between charged particles.
"""

import numpy as np
from typing import List
from . import constants as const
from .particle import Particle


def calculate_gravitational_force(particle: Particle) -> np.ndarray:
    """
    Calculate gravitational force on a particle.
    F_gravity = [0, m * g] where g is negative (downward)
    """
    force_x = 0.0
    force_y = particle.mass * const.GRAVITY
    return np.array([force_x, force_y])


def calculate_coulomb_force_between(particle1: Particle,
                                    particle2: Particle) -> np.ndarray:
    """
    Calculate electrostatic force between two charged particles.
    Uses a minimum distance cutoff to avoid singularity at r=0.
    """
    # Calculate displacement vector from particle2 to particle1
    displacement = particle1.position - particle2.position

    # Calculate distance between particles
    distance = np.linalg.norm(displacement)

    # Handle very close particles with stronger regularization
    min_distance = 1e-3  # Increased from 1e-6 for stronger regularization

    if distance < const.EPSILON:
        # For exactly overlapping particles, add small random force to break symmetry
        # This prevents particles from being stuck together
        np.random.seed(int(particle1.particle_id + particle2.particle_id))
        random_angle = np.random.uniform(0, 2 * np.pi)
        small_force = 0.1  # Small force to separate particles
        return np.array([small_force * np.cos(random_angle),
                        small_force * np.sin(random_angle)])
    elif distance < min_distance:
        # For very close but not overlapping, use minimum distance
        distance = min_distance
        r_hat = displacement / np.linalg.norm(displacement)
    else:
        # Normal case: calculate unit vector
        r_hat = displacement / distance

    # Calculate force magnitude with optional capping for numerical stability
    # F = q1 * q2 / r^2
    force_magnitude = (particle1.charge * particle2.charge) / (distance ** 2)

    # Cap force magnitude to prevent numerical issues
    max_force = 1e9  # Reasonable maximum force
    if force_magnitude > max_force:
        force_magnitude = max_force

    # Force vector points in direction of displacement (repulsive)
    force = force_magnitude * r_hat

    return force


def calculate_total_electrostatic_force(particle_index: int,
                                        particles: List[Particle]) -> np.ndarray:
    """
    Calculate total electrostatic force on one particle from all others.
    """
    total_force = np.array([0.0, 0.0])
    particle_i = particles[particle_index]

    for j, particle_j in enumerate(particles):
        if j == particle_index:
            continue

        pairwise_force = calculate_coulomb_force_between(particle_i, particle_j)
        total_force += pairwise_force

    return total_force


def calculate_total_force(particle_index: int,
                          particles: List[Particle]) -> np.ndarray:
    """
    Calculate total force on a particle (gravity + electrostatic).
    """
    particle = particles[particle_index]

    gravity_force = calculate_gravitational_force(particle)
    electrostatic_force = calculate_total_electrostatic_force(particle_index, particles)

    total_force = gravity_force + electrostatic_force
    return total_force


def calculate_acceleration(particle_index: int,
                           particles: List[Particle]) -> np.ndarray:
    """
    Calculate acceleration of a particle from total forces.
    Using Newton's second law: a = F/m
    """
    particle = particles[particle_index]
    total_force = calculate_total_force(particle_index, particles)

    # Handle zero mass case
    if abs(particle.mass) < const.EPSILON:
        return np.array([0.0, 0.0])

    # Divide force by mass for acceleration
    acceleration = total_force / particle.mass
    return acceleration


def calculate_potential_energy_coulomb(particles: List[Particle]) -> float:
    """
    Calculate total Coulomb potential energy of the system.
    U = (1/2) * sum_i sum_j (q_i * q_j / r_ij) for i != j
    """
    total_potential = 0.0

    # Sum over all unique pairs (i < j to avoid double counting)
    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            # Calculate distance between particles
            distance = particles[i].distance_to(particles[j])

            # Use same regularization as force calculation
            min_distance = 1e-3  # Match the force calculation
            if distance < min_distance:
                distance = min_distance

            # Add pairwise potential energy
            # U_ij = q_i * q_j / r_ij
            pair_potential = (particles[i].charge * particles[j].charge) / distance

            # Cap potential to prevent numerical issues
            max_potential = 1e9
            if pair_potential > max_potential:
                pair_potential = max_potential

            total_potential += pair_potential

    return total_potential


def calculate_system_forces(particles: List[Particle]) -> List[np.ndarray]:
    """
    Calculate forces on all particles in the system.

    This function calculates forces for all particles simultaneously,
    which is needed for the batch update approach in RK4 integration.
    """
    forces = []

    for i in range(len(particles)):
        force = calculate_total_force(i, particles)
        forces.append(force)

    return forces


def calculate_system_forces_symmetric(particles: List[Particle]) -> List[np.ndarray]:
    """
    Calculate forces on all particles ensuring Newton's 3rd law exactly.

    This version calculates each pair only once and applies equal and
    opposite forces, guaranteeing exact conservation of momentum.
    """
    n_particles = len(particles)
    forces = [np.array([0.0, 0.0]) for _ in range(n_particles)]

    # Add gravitational forces
    for i in range(n_particles):
        forces[i] += calculate_gravitational_force(particles[i])

    # Add Coulomb forces - calculate each pair only once
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            # Calculate force on i due to j
            force_on_i = calculate_coulomb_force_between(particles[i], particles[j])

            # Apply equal and opposite forces (Newton's 3rd law)
            forces[i] += force_on_i
            forces[j] -= force_on_i  # Equal and opposite

    return forces


def calculate_system_accelerations(particles: List[Particle]) -> List[np.ndarray]:
    """
    Calculate accelerations for all particles in the system.
    Uses symmetric force calculation for exact momentum conservation.
    """
    forces = calculate_system_forces_symmetric(particles)
    accelerations = []

    for i, particle in enumerate(particles):
        if abs(particle.mass) < const.EPSILON:
            accelerations.append(np.array([0.0, 0.0]))
        else:
            accelerations.append(forces[i] / particle.mass)

    return accelerations
