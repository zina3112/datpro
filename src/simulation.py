"""
simulation.py - Main simulation controller with EXACT collision handling

This module orchestrates the charged particle simulation using the
EXACT interpolation-based collision detection method as specified.
"""

import numpy as np
import os
from typing import List, Tuple, Optional
from . import constants as const
from .particle import Particle
from .box import Box
from .integrator import RK4Integrator, rk4_step_system
from .forces import (calculate_potential_energy_coulomb,
                     calculate_system_forces)
from .data_handler import DataHandler
import time as time_module


class Simulation:
    """
    Main simulation controller for charged particles in a box.

    This implementation uses the EXACT collision handling method specified:
    - Calculate RK4 steps for particles
    - Use linear interpolation to detect wall collisions
    - Split timesteps at collision points
    - Apply perfect elastic reflections
    """

    def __init__(self,
                 initial_states: Optional[np.ndarray] = None,
                 dt: float = const.DT,
                 output_file: str = None):
        """
        Initialize the simulation with particles and parameters.

        Args:
            initial_states: Initial state vectors [x, y, vx, vy] for each particle
            dt: Timestep for integration
            output_file: Path to output CSV file
        """
        if initial_states is None:
            initial_states = const.INITIAL_STATES

        # Initialize particles from initial states
        self.particles = []
        for i, state in enumerate(initial_states):
            particle = Particle(
                x=state[0], y=state[1],
                vx=state[2], vy=state[3],
                mass=const.MASS,
                charge=const.CHARGE
            )
            self.particles.append(particle)

        # Initialize simulation components
        self.box = Box()
        self.integrator = RK4Integrator(dt=dt)
        self.data_handler = DataHandler(output_file)

        # Simulation state
        self.current_time = 0.0
        self.step_count = 0
        self.dt = dt

        # Energy tracking
        self.initial_energy = self.calculate_total_energy()
        self.energy_history = [self.initial_energy]
        self.time_history = [0.0]

        # Record initial state
        self.data_handler.record_state(0.0, self.initial_energy, self.particles)

        # Performance tracking
        self.start_real_time = None
        self.total_computation_time = 0.0

        print(f"Simulation initialized with {len(self.particles)} particles")
        print(f"Initial total energy: {self.initial_energy:.6f}")
        print(f"Timestep: {self.dt}, Box: {self.box}")

    def calculate_total_energy(self) -> float:
        """
        Calculate the total energy of the system.

        Total energy = Kinetic + Gravitational Potential + Coulomb Potential

        Returns:
            float: Total system energy
        """
        total_energy = 0.0

        # Sum kinetic and gravitational potential energy for each particle
        for particle in self.particles:
            total_energy += particle.kinetic_energy()
            total_energy += particle.potential_energy_gravity()

        # Add Coulomb potential energy (counted once for all pairs)
        total_energy += calculate_potential_energy_coulomb(self.particles)

        return total_energy

    def step(self) -> bool:
        """
        Perform one simulation timestep using the EXACT collision handling method.

        This method implements the specification requirement:
        1. Calculate RK4 increments for all particles
        2. For each particle, check if it would leave the box
        3. If yes, use the exact interpolation method to handle collision
        4. Update all particles with their final states

        Returns:
            bool: True if step completed successfully
        """
        # CRITICAL: Store original states before any modifications
        # This is essential for the batch update approach
        original_states = [p.state.copy() for p in self.particles]

        # List to store the final states after collision handling
        final_states = []

        # STEP 1: Process each particle individually for collision detection
        # This follows the specification: "nacheinander für jedes Teilchen"
        for i, particle in enumerate(self.particles):
            # Use the EXACT interpolation method from the Box class
            # This method handles:
            # - Full RK4 calculation
            # - Collision detection via linear interpolation
            # - Timestep splitting at collision point
            # - Velocity reflection
            # - Continuation with remaining timestep

            final_state = self.box.handle_wall_collision_exact(
                particle,
                self.particles,
                i,
                self.dt
            )

            final_states.append(final_state)

        # STEP 2: Update all particles with their final states
        # This is done AFTER all calculations as specified:
        # "Statevektoren der Teilchen erst aktualisiert, nachdem diese
        # für alle Teilchen berechnet wurden"
        for particle, final_state in zip(self.particles, final_states):
            particle.update_state(final_state)

        # STEP 3: Safety check - ensure all particles are within bounds
        # This handles any numerical errors
        for particle in self.particles:
            self.box.enforce_boundaries(particle)

        # Update simulation time
        self.current_time += self.dt
        self.step_count += 1

        # Calculate and track energy
        current_energy = self.calculate_total_energy()
        self.energy_history.append(current_energy)
        self.time_history.append(self.current_time)

        # Check energy conservation (important for validation)
        if abs(self.initial_energy) > const.EPSILON:
            energy_drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)

            # Warn if energy drift exceeds 1%
            if energy_drift > 0.01:
                print(f"Warning: Energy drift = {energy_drift * 100:.2f}% at t={self.current_time:.3f}")

            # Error if energy drift exceeds 10% (indicates serious numerical issues)
            if energy_drift > 0.1:
                print(f"ERROR: Excessive energy drift = {energy_drift * 100:.2f}%")
                print("Check timestep size or collision handling")

        # Record data for output
        self.data_handler.record_state(
            self.current_time,
            current_energy,
            self.particles
        )

        return True

    def step_alternative_batch(self) -> bool:
        """
        Alternative implementation using batch RK4 followed by collision handling.

        This method can be used for comparison but the main step() method
        follows the specification more closely.
        """
        # Store original states
        original_states = [p.state.copy() for p in self.particles]

        # Calculate RK4 increments for all particles at once
        state_increments = rk4_step_system(self.particles, self.dt)

        # Apply increments and handle collisions
        for i, (particle, increment) in enumerate(zip(self.particles, state_increments)):
            tentative_state = original_states[i] + increment

            # Check if particle would be outside box
            if not self.box.is_inside(tentative_state[0:2]):
                # Use exact collision handling
                final_state = self.box.handle_wall_collision_exact(
                    particle,
                    self.particles,
                    i,
                    self.dt
                )
                particle.update_state(final_state)
            else:
                # No collision - use tentative state
                particle.update_state(tentative_state)

        # Update time and record data
        self.current_time += self.dt
        self.step_count += 1

        current_energy = self.calculate_total_energy()
        self.energy_history.append(current_energy)
        self.time_history.append(self.current_time)

        self.data_handler.record_state(
            self.current_time,
            current_energy,
            self.particles
        )

        return True

    def run(self,
            simulation_time: float = const.SIMULATION_TIME,
            progress_interval: int = 1000) -> None:
        """
        Run the complete simulation for the specified time.

        Args:
            simulation_time: Total time to simulate (seconds)
            progress_interval: Steps between progress updates
        """
        print(f"\nStarting simulation for {simulation_time} seconds...")
        print(f"Using EXACT interpolation method for wall collisions")

        # Validate parameters
        if abs(self.dt) < const.EPSILON:
            print("Error: Timestep is zero or too small")
            return

        if abs(simulation_time) < const.EPSILON:
            print("Simulation time is zero - saving initial state only")
            self.print_statistics()
            self.data_handler.save()
            print(f"\nData saved to {self.data_handler.output_file}")
            return

        # Calculate total number of steps
        total_steps = int(abs(simulation_time / self.dt))
        print(f"Total steps: {total_steps}")
        print(f"Energy tolerance: {const.ENERGY_TOLERANCE}")

        # Start timing
        self.start_real_time = time_module.time()

        # Main simulation loop
        step_counter = 0
        while step_counter < total_steps:
            # Perform one timestep
            success = self.step()

            if not success:
                print("Simulation stopped early due to error")
                break

            step_counter += 1

            # Progress update
            if step_counter % progress_interval == 0:
                self._print_progress(step_counter, total_steps)

        # Calculate total computation time
        self.total_computation_time = time_module.time() - self.start_real_time

        # Print final statistics
        self.print_statistics()

        # Save data to file
        self.data_handler.save()
        print(f"\nData saved to {self.data_handler.output_file}")

    def _print_progress(self, current_step: int, total_steps: int):
        """
        Print progress information during simulation.

        Args:
            current_step: Current step number
            total_steps: Total number of steps
        """
        progress = (current_step / total_steps) * 100

        # Calculate energy drift
        if abs(self.initial_energy) > const.EPSILON:
            energy_drift = abs(self.energy_history[-1] - self.initial_energy) / abs(self.initial_energy)
        else:
            energy_drift = 0.0

        # Calculate performance
        elapsed_real_time = time_module.time() - self.start_real_time
        steps_per_second = current_step / elapsed_real_time if elapsed_real_time > 0 else 0

        # Count total collisions
        total_collisions = sum(p.collision_count for p in self.particles)

        print(f"Progress: {progress:.1f}% | "
              f"Time: {self.current_time:.3f}s | "
              f"Energy drift: {energy_drift * 100:.4f}% | "
              f"Collisions: {total_collisions} | "
              f"Performance: {steps_per_second:.0f} steps/s")

    def print_statistics(self):
        """Print comprehensive simulation statistics."""
        print("\n" + "=" * 60)
        print("SIMULATION STATISTICS")
        print("=" * 60)

        # Time statistics
        print(f"\nTime Statistics:")
        print(f"  Simulation time: {self.current_time:.3f} seconds")
        print(f"  Number of steps: {self.step_count}")
        print(f"  Timestep size: {self.dt}")
        print(f"  Computation time: {self.total_computation_time:.2f} seconds")

        if self.total_computation_time > 0:
            speedup = abs(self.current_time) / self.total_computation_time
            print(f"  Speed ratio: {speedup:.2f}x real-time")

        # Energy conservation statistics
        final_energy = self.energy_history[-1] if self.energy_history else self.initial_energy
        energy_drift = abs(final_energy - self.initial_energy)

        if abs(self.initial_energy) > const.EPSILON:
            relative_drift = energy_drift / abs(self.initial_energy)
        else:
            relative_drift = 0.0

        print(f"\nEnergy Conservation:")
        print(f"  Initial energy: {self.initial_energy:.6f}")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Absolute drift: {energy_drift:.6e}")
        print(f"  Relative drift: {relative_drift * 100:.4f}%")

        # Check if energy conservation is within tolerance
        if relative_drift < const.ENERGY_TOLERANCE:
            print(f"  ✓ Energy conserved within tolerance ({const.ENERGY_TOLERANCE * 100:.4f}%)")
        else:
            print(f"  ✗ Energy drift exceeds tolerance ({const.ENERGY_TOLERANCE * 100:.4f}%)")

        # Collision statistics
        total_collisions = sum(p.collision_count for p in self.particles)
        print(f"\nCollision Statistics (using EXACT interpolation):")
        print(f"  Total wall collisions: {total_collisions}")
        print(f"  Box collision counter: {self.box.total_collisions}")

        for i, particle in enumerate(self.particles):
            if particle.collision_count > 0:
                print(f"  Particle {i + 1}: {particle.collision_count} collisions")

        # Particle final states
        print(f"\nFinal Particle States:")
        for i, particle in enumerate(self.particles):
            print(f"  Particle {i + 1}: pos=({particle.x:.2f}, {particle.y:.2f}), "
                  f"vel=({particle.vx:.2f}, {particle.vy:.2f})")

        print("=" * 60)

    def get_trajectory(self, particle_index: int) -> Tuple[List[float], List[float]]:
        """
        Get the trajectory of a specific particle.

        Args:
            particle_index: Index of the particle (0-based)

        Returns:
            Tuple of (x_positions, y_positions)
        """
        return self.data_handler.get_particle_trajectory(particle_index)

    def get_energy_history(self) -> Tuple[List[float], List[float]]:
        """
        Get the energy history of the simulation.

        Returns:
            Tuple of (times, energies)
        """
        return self.time_history, self.energy_history
