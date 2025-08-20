"""
simulation.py - Main simulation controller

This module orchestrates the complete charged particle simulation.
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
    """

    def __init__(self,
                 initial_states: Optional[np.ndarray] = None,
                 dt: float = const.DT,
                 output_file: str = None):
        """Initialize the simulation."""
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

        self.box = Box()
        self.integrator = RK4Integrator(dt=dt)
        self.data_handler = DataHandler(output_file)

        self.current_time = 0.0
        self.step_count = 0
        self.dt = dt

        self.initial_energy = self.calculate_total_energy()
        self.energy_history = [self.initial_energy]
        self.time_history = [0.0]

        # Record initial state
        self.data_handler.record_state(0.0, self.initial_energy, self.particles)

        self.start_real_time = None
        self.total_computation_time = 0.0

        print(f"Simulation initialized with {len(self.particles)} particles")
        print(f"Initial total energy: {self.initial_energy:.6f}")
        print(f"Timestep: {self.dt}, Box: {self.box}")

    def calculate_total_energy(self) -> float:
        """Calculate the total energy of the system."""
        total_energy = 0.0

        # Sum kinetic and gravitational potential energy for each particle
        for particle in self.particles:
            total_energy += particle.kinetic_energy()
            total_energy += particle.potential_energy_gravity()

        # Add Coulomb potential energy
        total_energy += calculate_potential_energy_coulomb(self.particles)

        return total_energy

    def step(self) -> bool:
        """
        Perform one simulation timestep with improved collision handling.
        """
        # Store original states
        original_states = [p.state.copy() for p in self.particles]

        # Calculate RK4 increments for all particles
        state_increments = rk4_step_system(self.particles, self.dt)

        # Update particles and handle collisions
        for i, (particle, increment) in enumerate(zip(self.particles, state_increments)):
            new_state = original_states[i] + increment
            particle.update_state(new_state)

            # Check and handle collisions after update
            self.box.check_and_handle_collisions(particle, self.dt)

        # Update simulation time
        self.current_time += self.dt
        self.step_count += 1

        # Calculate and track energy
        current_energy = self.calculate_total_energy()
        self.energy_history.append(current_energy)
        self.time_history.append(self.current_time)

        # Check energy conservation
        if abs(self.initial_energy) > const.EPSILON:
            energy_drift = abs(current_energy - self.initial_energy) / abs(self.initial_energy)
            if energy_drift > 0.01:  # 1% drift warning
                print(f"Warning: Energy drift = {energy_drift * 100:.2f}% at t={self.current_time:.3f}")

        # Record data
        self.data_handler.record_state(
            self.current_time,
            current_energy,
            self.particles
        )

        return True

    def run(self,
            simulation_time: float = const.SIMULATION_TIME,
            progress_interval: int = 1000) -> None:
        """Run the complete simulation."""
        print(f"\nStarting simulation for {simulation_time} seconds...")

        if abs(self.dt) < const.EPSILON:
            print("Error: Timestep is zero or too small")
            return

        if abs(simulation_time) < const.EPSILON:
            self.print_statistics()
            self.data_handler.save()
            print(f"\nData saved to {self.data_handler.output_file}")
            return

        total_steps = int(abs(simulation_time / self.dt))
        print(f"Total steps: {total_steps}")

        self.start_real_time = time_module.time()

        step_counter = 0
        while step_counter < total_steps:
            success = self.step()

            if not success:
                print("Simulation stopped early due to error")
                break

            step_counter += 1

            # Progress update
            if step_counter % progress_interval == 0:
                progress = (step_counter / total_steps) * 100

                if abs(self.initial_energy) > const.EPSILON:
                    energy_drift = abs(self.energy_history[-1] - self.initial_energy) / abs(self.initial_energy)
                else:
                    energy_drift = 0.0

                elapsed_real_time = time_module.time() - self.start_real_time
                steps_per_second = step_counter / elapsed_real_time if elapsed_real_time > 0 else 0

                print(f"Progress: {progress:.1f}% | "
                      f"Time: {self.current_time:.3f}s | "
                      f"Energy drift: {energy_drift * 100:.4f}% | "
                      f"Performance: {steps_per_second:.0f} steps/s")

        self.total_computation_time = time_module.time() - self.start_real_time
        self.print_statistics()

        self.data_handler.save()
        print(f"\nData saved to {self.data_handler.output_file}")

    def print_statistics(self):
        """Print simulation statistics."""
        print("\n" + "=" * 60)
        print("SIMULATION STATISTICS")
        print("=" * 60)

        print(f"Simulation time: {self.current_time:.3f} seconds")
        print(f"Number of steps: {self.step_count}")
        print(f"Computation time: {self.total_computation_time:.2f} seconds")

        if self.total_computation_time > 0:
            speedup = abs(self.current_time) / self.total_computation_time
            print(f"Speed ratio: {speedup:.2f}x real-time")

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

        total_collisions = sum(p.collision_count for p in self.particles)
        print(f"\nCollision Statistics:")
        print(f"  Total wall collisions: {total_collisions}")

        for i, particle in enumerate(self.particles):
            if particle.collision_count > 0:
                print(f"  Particle {i + 1}: {particle.collision_count} collisions")

        print("=" * 60)

    def get_trajectory(self, particle_index: int) -> Tuple[List[float], List[float]]:
        """Get the trajectory of a specific particle."""
        return self.data_handler.get_particle_trajectory(particle_index)

    def get_energy_history(self) -> Tuple[List[float], List[float]]:
        """Get the energy history of the simulation."""
        return self.time_history, self.energy_history
