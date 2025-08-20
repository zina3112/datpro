#!/usr/bin/env python3
"""
main.py - Main execution script for charged particle simulation

This script runs the complete charged particle simulation with the
specified initial conditions and generates all required outputs.

Usage:
    python main.py [options]

Options:
    --dt TIMESTEP       Set integration timestep (default: 0.001)
    --time TIME         Set simulation time in seconds (default: 10.0)
    --output FILE       Set output file path
    --no-plots          Skip generating plots
    --test              Run with test configuration

Author: Simulation Team
Date: 2024
"""

from src.simulation import Simulation
from src.visualization import Visualizer
import src.constants as const
import sys
import os
import argparse
import numpy as np
import time

# Add parent directory to path so we can import src as a package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Now import from src package


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='Run charged particle simulation in 2D box'
    )

    parser.add_argument(
        '--dt', type=float, default=const.DT,
        help=f'Integration timestep (default: {const.DT})'
    )

    parser.add_argument(
        '--time', type=float, default=const.SIMULATION_TIME,
        help=f'Simulation time in seconds (default: {const.SIMULATION_TIME})'
    )

    parser.add_argument(
        '--output', type=str, default=None,
        help='Output file path (default: auto-generated)'
    )

    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )

    parser.add_argument(
        '--test', action='store_true',
        help='Run with test configuration (shorter simulation)'
    )

    parser.add_argument(
        '--progress', type=int, default=1000,
        help='Progress update interval in steps (default: 1000)'
    )

    return parser.parse_args()


def print_header():
    """Print simulation header information."""
    print("=" * 70)
    print("CHARGED PARTICLE SIMULATION IN 2D BOX")
    print("=" * 70)
    print()
    print("Simulation Parameters:")
    print(f"  - Number of particles: {const.N_PARTICLES}")
    print(f"  - Particle mass: {const.MASS}")
    print(f"  - Particle charge: {const.CHARGE}")
    print(f"  - Gravity: {const.GRAVITY}")
    print(f"  - Box dimensions: [{const.BOX_MIN_X}, {const.BOX_MAX_X}] × "
          f"[{const.BOX_MIN_Y}, {const.BOX_MAX_Y}]")
    print()
    print("Initial Particle States (x, y, vx, vy):")
    for i, state in enumerate(const.INITIAL_STATES):
        print(f"  Particle {i + 1}: {state}")
    print()
    print("=" * 70)
    print()


def run_tests():
    """
    Run unit tests.

    Returns:
        bool: True if all tests pass
    """
    print("Running unit tests...")
    import unittest

    # Discover and run tests
    loader = unittest.TestLoader()
    # FIX: Use the correct test directory path
    test_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')

    # Alternative if tests are in same directory as src
    if not os.path.exists(test_dir):
        test_dir = os.path.join(os.path.dirname(__file__), '..', 'tests')

    suite = loader.discover(test_dir, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Print header
    print_header()

    # Run tests if requested
    if args.test:
        print("TEST MODE: Running unit tests first...")
        if not run_tests():
            print("ERROR: Unit tests failed. Exiting.")
            return 1
        print("\nUnit tests passed. Proceeding with simulation...\n")

        # Use shorter simulation for test mode
        simulation_time = 1.0
        print(f"TEST MODE: Running shortened simulation ({simulation_time}s)")
    else:
        simulation_time = args.time

    # Validate timestep
    if abs(args.dt) < 1e-10 and args.dt != 0:
        print(f"Warning: Timestep {args.dt} is very small, using 0.001 instead")
        dt = 0.001
    else:
        dt = args.dt

    # Create output filename with timestamp if not specified
    if args.output is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(
            const.OUTPUT_DIR,
            f"simulation_{timestamp}.csv"
        )
    else:
        output_file = args.output

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    print(f"Configuration:")
    print(f"  - Timestep: {dt}")
    print(f"  - Simulation time: {simulation_time}")
    print(f"  - Output file: {output_file}")
    print(f"  - Generate plots: {not args.no_plots}")
    print()

    # Create and run simulation
    print("Initializing simulation...")
    sim = Simulation(
        initial_states=const.INITIAL_STATES,
        dt=dt,
        output_file=output_file
    )

    # Run simulation
    sim.run(
        simulation_time=simulation_time,
        progress_interval=args.progress
    )

    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")

        visualizer = Visualizer(sim.data_handler)

        # Task 5: Plot energy vs time
        print("  - Creating energy conservation plot...")
        visualizer.plot_energy_vs_time(save=True, show=False)

        # Task 6: Plot trajectory of first particle
        print("  - Creating trajectory plot for Particle 1...")
        visualizer.plot_particle_trajectory(0, save=True, show=False)

        # Task 7: Plot all trajectories
        print("  - Creating combined trajectory plot...")
        visualizer.plot_all_trajectories(save=True, show=False)

        # Additional visualizations
        print("  - Creating complete visualization report...")
        visualizer.create_summary_report()

        print(f"\nPlots saved to: {visualizer.figure_dir}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    # Print final summary
    stats = sim.data_handler.get_statistics()
    print("\nFinal Statistics:")
    print(f"  - Total timesteps: {stats['num_timesteps']}")
    print(f"  - Initial energy: {stats['initial_energy']:.6f}")
    print(f"  - Final energy: {stats['final_energy']:.6f}")
    print(f"  - Energy drift: {stats['energy_drift']:.6e}")
    print(f"  - Relative drift: {stats['relative_drift'] * 100:.4f}%")

    # Success message
    print("\n✓ All tasks completed successfully!")
    print(f"✓ Results saved to: {output_file}")
    if not args.no_plots:
        print(f"✓ Plots saved to: {visualizer.figure_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
