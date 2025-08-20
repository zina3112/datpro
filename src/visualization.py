"""
visualization.py - Plotting and visualization module

This module provides visualization capabilities for the simulation,
including energy plots, particle trajectories, and animation.

Author: Simulation Team
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import os
from . import constants as const
from .data_handler import DataHandler


class Visualizer:
    """
    Visualization tools for the charged particle simulation.

    Provides methods for:
    - Energy conservation plots
    - Individual particle trajectories
    - Combined trajectory plots
    - Phase space diagrams

    Attributes:
        data_handler: DataHandler object with simulation data
        figure_dir: Directory for saving figures
    """

    def __init__(self, data_handler: DataHandler, figure_dir: Optional[str] = None):
        """
        Initialize the visualizer.

        Args:
            data_handler: DataHandler with simulation data
            figure_dir: Directory for saving figures
        """
        self.data_handler = data_handler

        # Set figure directory
        if figure_dir is None:
            figure_dir = os.path.join(const.OUTPUT_DIR, const.PLOT_DIR)
        self.figure_dir = figure_dir

        # Create directory if it doesn't exist
        os.makedirs(self.figure_dir, exist_ok=True)

        # Set matplotlib style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Fall back to default if style not available
            pass

    def plot_energy_vs_time(self, save: bool = True, show: bool = True) -> None:
        """
        Plot total energy versus time to check conservation.

        Args:
            save: Whether to save the figure
            show: Whether to display the figure
        """
        times, energies = self.data_handler.get_energy_history()

        if not times:
            print("No data to plot")
            return

        # Calculate energy drift
        initial_energy = energies[0]
        relative_drift = [(e - initial_energy) / abs(initial_energy) * 100
                          for e in energies]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot absolute energy
        ax1.plot(times, energies, 'b-', linewidth=1.5, label='Total Energy')
        ax1.axhline(y=initial_energy, color='r', linestyle='--',
                    alpha=0.5, label=f'Initial Energy = {initial_energy:.4f}')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Total Energy')
        ax1.set_title('Total System Energy vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot relative energy drift
        ax2.plot(times, relative_drift, 'r-', linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Energy Drift (%)')
        ax2.set_title('Relative Energy Drift')
        ax2.grid(True, alpha=0.3)

        # Add final drift annotation
        final_drift = relative_drift[-1]
        ax2.text(0.98, 0.98, f'Final drift: {final_drift:.4f}%',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save:
            filename = os.path.join(self.figure_dir, 'energy_conservation.png')
            plt.savefig(filename, dpi=const.DPI, bbox_inches='tight')
            print(f"Energy plot saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_particle_trajectory(self,
                                 particle_index: int,
                                 save: bool = True,
                                 show: bool = True) -> None:
        """
        Plot the trajectory of a single particle.

        Args:
            particle_index: Index of particle to plot (0-based)
            save: Whether to save the figure
            show: Whether to display the figure
        """
        # FIX: Handle invalid index gracefully
        try:
            x_positions, y_positions = self.data_handler.get_particle_trajectory(particle_index)
        except (ValueError, IndexError) as e:
            print(f"Cannot plot trajectory: {e}")
            return

        if not x_positions:
            print("No trajectory data to plot")
            return

        fig, ax = plt.subplots(figsize=const.FIGURE_SIZE)

        # Plot trajectory
        ax.plot(x_positions, y_positions, 'b-', linewidth=1, alpha=0.7)

        # Mark start and end points
        ax.plot(x_positions[0], y_positions[0], 'go',
                markersize=10, label='Start', markeredgecolor='darkgreen')
        ax.plot(x_positions[-1], y_positions[-1], 'ro',
                markersize=10, label='End', markeredgecolor='darkred')

        # Draw box boundaries
        self._draw_box_boundaries(ax)

        # Labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Trajectory of Particle {particle_index + 1}')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if save:
            filename = os.path.join(self.figure_dir,
                                    f'trajectory_particle_{particle_index + 1}.png')
            plt.savefig(filename, dpi=const.DPI, bbox_inches='tight')
            print(f"Trajectory plot saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_all_trajectories(self, save: bool = True, show: bool = True) -> None:
        """
        Plot trajectories of all particles in the same figure.

        Args:
            save: Whether to save the figure
            show: Whether to display the figure
        """
        trajectories = self.data_handler.get_all_trajectories()

        if not trajectories:
            print("No trajectory data to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Color map for different particles
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))

        # Plot each particle's trajectory
        for i, (x_pos, y_pos) in enumerate(trajectories):
            if not x_pos:  # Skip empty trajectories
                continue

            ax.plot(x_pos, y_pos, color=colors[i], linewidth=1,
                    alpha=0.6, label=f'Particle {i + 1}')

            # Mark start positions
            ax.plot(x_pos[0], y_pos[0], 'o', color=colors[i],
                    markersize=8, markeredgecolor='black')

            # Mark end positions
            ax.plot(x_pos[-1], y_pos[-1], 's', color=colors[i],
                    markersize=8, markeredgecolor='black')

        # Draw box boundaries
        self._draw_box_boundaries(ax)

        # Labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Trajectories of All Particles')
        if len(trajectories) <= 10:  # Only show legend for reasonable number of particles
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filename = os.path.join(self.figure_dir, 'all_trajectories.png')
            plt.savefig(filename, dpi=const.DPI, bbox_inches='tight')
            print(f"Combined trajectory plot saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_phase_space(self,
                         particle_index: int,
                         save: bool = True,
                         show: bool = True) -> None:
        """
        Plot phase space diagram for a particle.

        Args:
            particle_index: Index of particle
            save: Whether to save the figure
            show: Whether to display the figure
        """
        # Get particle data
        if particle_index >= const.N_PARTICLES:
            print(f"Invalid particle index: {particle_index}")
            return

        particle_data = []
        for timestep in self.data_handler.trajectory_data['particles']:
            if particle_index < len(timestep):
                particle_data.append(timestep[particle_index])

        if not particle_data:
            print("No data for phase space plot")
            return

        # Extract positions and velocities
        x_positions = [p['x'] for p in particle_data]
        y_positions = [p['y'] for p in particle_data]
        vx_velocities = [p['vx'] for p in particle_data]
        vy_velocities = [p['vy'] for p in particle_data]

        # Create phase space plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # X phase space
        ax1.plot(x_positions, vx_velocities, 'b-', linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('X Velocity')
        ax1.set_title('X Phase Space')
        ax1.grid(True, alpha=0.3)

        # Y phase space
        ax2.plot(y_positions, vy_velocities, 'r-', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('Y Position')
        ax2.set_ylabel('Y Velocity')
        ax2.set_title('Y Phase Space')
        ax2.grid(True, alpha=0.3)

        # Position space
        ax3.plot(x_positions, y_positions, 'g-', linewidth=0.5, alpha=0.7)
        ax3.set_xlabel('X Position')
        ax3.set_ylabel('Y Position')
        ax3.set_title('Position Space')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)

        # Velocity space
        ax4.plot(vx_velocities, vy_velocities, 'm-', linewidth=0.5, alpha=0.7)
        ax4.set_xlabel('X Velocity')
        ax4.set_ylabel('Y Velocity')
        ax4.set_title('Velocity Space')
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Phase Space Diagrams - Particle {particle_index + 1}')
        plt.tight_layout()

        if save:
            filename = os.path.join(self.figure_dir,
                                    f'phase_space_particle_{particle_index + 1}.png')
            plt.savefig(filename, dpi=const.DPI, bbox_inches='tight')
            print(f"Phase space plot saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close()

    def _draw_box_boundaries(self, ax) -> None:
        """
        Draw box boundaries on the plot.

        Args:
            ax: Matplotlib axis object
        """
        # Draw box walls
        box_x = [const.BOX_MIN_X, const.BOX_MAX_X, const.BOX_MAX_X,
                 const.BOX_MIN_X, const.BOX_MIN_X]
        box_y = [const.BOX_MIN_Y, const.BOX_MIN_Y, const.BOX_MAX_Y,
                 const.BOX_MAX_Y, const.BOX_MIN_Y]

        ax.plot(box_x, box_y, 'k-', linewidth=2, label='Box boundary')

        # Set axis limits with small margin
        margin = 5
        ax.set_xlim(const.BOX_MIN_X - margin, const.BOX_MAX_X + margin)
        ax.set_ylim(const.BOX_MIN_Y - margin, const.BOX_MAX_Y + margin)

    def create_summary_report(self) -> None:
        """Create a comprehensive summary report with all plots."""
        print("\nGenerating summary report...")

        # Energy conservation plot
        self.plot_energy_vs_time(save=True, show=False)

        # Individual particle trajectories
        for i in range(const.N_PARTICLES):
            self.plot_particle_trajectory(i, save=True, show=False)
            self.plot_phase_space(i, save=True, show=False)

        # Combined trajectories
        self.plot_all_trajectories(save=True, show=False)

        # Statistics
        stats = self.data_handler.get_statistics()

        # Write statistics to file
        stats_file = os.path.join(self.figure_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("SIMULATION STATISTICS\n")
            f.write("=" * 50 + "\n\n")

            for key, value in stats.items():
                f.write(f"{key}: {value}\n")

        print(f"Summary report saved to {self.figure_dir}")
