"""
constants.py - Physical and numerical constants for charged particle simulation

This module contains all physical constants and simulation parameters used
throughout the charged particle simulation. Centralizing constants ensures
consistency and makes parameter tuning easier.

Author: Simulation Team
Date: 2024
"""

import numpy as np

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

# Particle properties
MASS = 1.0  # Mass of each particle (dimensionless units)
CHARGE = 50.0  # Charge of each particle (dimensionless units)

# External forces
GRAVITY = -10.0  # Gravitational acceleration in y-direction (negative = downward)

# ============================================================================
# SIMULATION BOX PARAMETERS
# ============================================================================

# Box dimensions (square box)
BOX_MIN_X = 0.0  # Minimum x-coordinate of the box
BOX_MAX_X = 100.0  # Maximum x-coordinate of the box
BOX_MIN_Y = 0.0  # Minimum y-coordinate of the box
BOX_MAX_Y = 100.0  # Maximum y-coordinate of the box

# Convenience box size variables
BOX_WIDTH = BOX_MAX_X - BOX_MIN_X
BOX_HEIGHT = BOX_MAX_Y - BOX_MIN_Y

# ============================================================================
# NUMERICAL PARAMETERS
# ============================================================================

# Time integration parameters
DT = 0.001  # Time step size in seconds
SIMULATION_TIME = 10.0  # Total simulation time in seconds
N_STEPS = int(SIMULATION_TIME / DT)  # Total number of time steps

# Numerical tolerance for floating point comparisons
EPSILON = 1e-10  # Small value for numerical comparisons
ENERGY_TOLERANCE = 1e-6  # Tolerance for energy conservation checks

# Wall collision detection parameters
COLLISION_EPSILON = 1e-10  # Tolerance for wall collision detection (same as EPSILON for consistency)
MAX_COLLISION_ITERATIONS = 10  # Maximum iterations for collision handling

# ============================================================================
# INITIAL CONDITIONS
# ============================================================================

# Initial state vectors for 7 particles
# Format: [x, y, vx, vy] for each particle
INITIAL_STATES = np.array([
    [1.0, 45.0, 10.0, 0.0],    # Particle 1: Moving right from left side
    [99.0, 55.0, -10.0, 0.0],  # Particle 2: Moving left from right side
    [10.0, 50.0, 15.0, -15.0],  # Particle 3: Diagonal motion
    [20.0, 30.0, -15.0, -15.0],  # Particle 4: Diagonal motion
    [80.0, 70.0, 15.0, 15.0],   # Particle 5: Diagonal motion upward
    [80.0, 60.0, 15.0, 15.0],   # Particle 6: Same velocity as 5, different position
    [80.0, 50.0, 15.0, 15.0]    # Particle 7: Same velocity as 5&6, different position
])

# Number of particles in the simulation
N_PARTICLES = len(INITIAL_STATES)

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Output file settings
OUTPUT_DIR = "output"
OUTPUT_FILE = "simulation_results.csv"
PLOT_DIR = "plots"

# Output frequency (save every nth step, 1 = save all steps)
OUTPUT_FREQUENCY = 1

# Plotting parameters
FIGURE_SIZE = (12, 8)  # Default figure size for plots
DPI = 100  # Resolution for saved figures
