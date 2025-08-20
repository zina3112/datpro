"""
data_handler.py - Data output and management module

This module handles all data recording and file I/O for the simulation,
including writing CSV output files and managing trajectory data.

Author: Simulation Team
Date: 2024
"""

import numpy as np
import os
from typing import List, Tuple, Optional
import csv
from . import constants as const
from .particle import Particle


class DataHandler:
    """
    Manages data recording and output for the simulation.

    This class handles:
    - Recording particle states at each timestep
    - Writing data to CSV files
    - Managing trajectory history
    - Providing data access for visualization

    Attributes:
        output_file: Path to output CSV file
        data_buffer: Buffer for storing simulation data
        trajectory_data: Complete trajectory history
    """

    def __init__(self, output_file: Optional[str] = None):
        """
        Initialize the data handler.

        Args:
            output_file: Path to output file (uses default if None)
        """
        # Set output file path
        if output_file is None:
            # Create output directory if it doesn't exist
            os.makedirs(const.OUTPUT_DIR, exist_ok=True)
            output_file = os.path.join(const.OUTPUT_DIR, const.OUTPUT_FILE)

        self.output_file = output_file

        # Initialize data storage
        self.data_buffer = []
        self.trajectory_data = {
            'time': [],
            'energy': [],
            'particles': []
        }

        # CSV header
        self.header = self._generate_header()

        # Statistics
        self.records_written = 0

        print(f"DataHandler initialized. Output file: {self.output_file}")

    def _generate_header(self) -> List[str]:
        """
        Generate CSV header based on number of particles.

        Format: t, E_total, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...

        Returns:
            List[str]: Header column names
        """
        header = ['t', 'E_total']

        for i in range(const.N_PARTICLES):
            particle_num = i + 1
            header.extend([
                f'x{particle_num}',
                f'y{particle_num}',
                f'vx{particle_num}',
                f'vy{particle_num}'
            ])

        return header

    def record_state(self,
                     time: float,
                     total_energy: float,
                     particles: List[Particle]) -> None:
        """
        Record the current state of the simulation.

        Args:
            time: Current simulation time
            total_energy: Total system energy
            particles: List of all particles
        """
        # Create data row
        row = [time, total_energy]

        # Add particle states
        for particle in particles:
            row.extend([
                particle.x,
                particle.y,
                particle.vx,
                particle.vy
            ])

        # Add to buffer
        self.data_buffer.append(row)

        # Store in trajectory data for easy access
        self.trajectory_data['time'].append(time)
        self.trajectory_data['energy'].append(total_energy)

        # Store particle states
        particle_states = []
        for particle in particles:
            particle_states.append({
                'x': particle.x,
                'y': particle.y,
                'vx': particle.vx,
                'vy': particle.vy
            })
        self.trajectory_data['particles'].append(particle_states)

        self.records_written += 1

    def save(self, filename: Optional[str] = None) -> None:
        """
        Save all recorded data to CSV file.

        Args:
            filename: Optional alternative filename
        """
        output_file = filename if filename else self.output_file

        try:
            with open(output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(self.header)

                # Write all data rows
                writer.writerows(self.data_buffer)

            print(f"Successfully saved {self.records_written} records to {output_file}")

        except IOError as e:
            print(f"Error saving data: {e}")

    def save_incremental(self) -> None:
        """
        Save data incrementally (append mode).

        Useful for long simulations to avoid data loss.
        """
        # Check if file exists to determine if we need header
        write_header = not os.path.exists(self.output_file)

        try:
            with open(self.output_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header if new file
                if write_header:
                    writer.writerow(self.header)

                # Write buffered data
                writer.writerows(self.data_buffer)

            # Clear buffer after writing
            self.data_buffer = []

        except IOError as e:
            print(f"Error in incremental save: {e}")

    def get_particle_trajectory(self, particle_index: int) -> Tuple[List[float], List[float]]:
        """
        Get the trajectory of a specific particle.

        Args:
            particle_index: Index of the particle (0-based)

        Returns:
            Tuple of (x_positions, y_positions)
        """
        if particle_index >= const.N_PARTICLES:
            raise ValueError(f"Particle index {particle_index} out of range")

        x_positions = []
        y_positions = []

        for timestep_data in self.trajectory_data['particles']:
            particle_state = timestep_data[particle_index]
            x_positions.append(particle_state['x'])
            y_positions.append(particle_state['y'])

        return x_positions, y_positions

    def get_all_trajectories(self) -> List[Tuple[List[float], List[float]]]:
        """
        Get trajectories for all particles.

        Returns:
            List of (x_positions, y_positions) for each particle
        """
        trajectories = []

        for i in range(const.N_PARTICLES):
            trajectory = self.get_particle_trajectory(i)
            trajectories.append(trajectory)

        return trajectories

    def get_energy_history(self) -> Tuple[List[float], List[float]]:
        """
        Get the energy history.

        Returns:
            Tuple of (times, energies)
        """
        return self.trajectory_data['time'], self.trajectory_data['energy']

    def load_from_file(self, filename: str) -> None:
        """
        Load simulation data from a CSV file.

        Args:
            filename: Path to CSV file
        """
        self.trajectory_data = {
            'time': [],
            'energy': [],
            'particles': []
        }

        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    # Extract time and energy
                    self.trajectory_data['time'].append(float(row['t']))
                    self.trajectory_data['energy'].append(float(row['E_total']))

                    # Extract particle states
                    particle_states = []
                    for i in range(const.N_PARTICLES):
                        p_num = i + 1
                        state = {
                            'x': float(row[f'x{p_num}']),
                            'y': float(row[f'y{p_num}']),
                            'vx': float(row[f'vx{p_num}']),
                            'vy': float(row[f'vy{p_num}'])
                        }
                        particle_states.append(state)

                    self.trajectory_data['particles'].append(particle_states)

            print(f"Loaded {len(self.trajectory_data['time'])} timesteps from {filename}")

        except IOError as e:
            print(f"Error loading data: {e}")

    def get_statistics(self) -> dict:
        """
        Calculate statistics from recorded data.

        Returns:
            dict: Statistics including energy drift, etc.
        """
        if not self.trajectory_data['energy']:
            return {}

        energies = np.array(self.trajectory_data['energy'])

        stats = {
            'initial_energy': energies[0],
            'final_energy': energies[-1],
            'mean_energy': np.mean(energies),
            'std_energy': np.std(energies),
            'max_energy': np.max(energies),
            'min_energy': np.min(energies),
            'energy_drift': energies[-1] - energies[0],
            'relative_drift': (energies[-1] - energies[0]) / abs(energies[0]),
            'num_timesteps': len(energies)
        }

        return stats
