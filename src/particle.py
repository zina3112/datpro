"""
particle.py - Particle class for charged particle simulation

This module defines the Particle class which represents a single charged
particle in the simulation. Each particle has position, velocity, mass,
and charge properties.

The state vector representation [x, y, vx, vy] is used for numerical
integration with the RK4 method.

Author: Simulation Team
Date: 2024
"""

import numpy as np
from typing import Tuple, Optional
from . import constants as const


class Particle:
    """
    Represents a charged particle in 2D space.

    The particle has mass, charge, position, and velocity. The state
    is represented as a 4D vector [x, y, vx, vy] for integration purposes.

    Attributes:
        mass (float): Mass of the particle (default from constants)
        charge (float): Charge of the particle (default from constants)
        state (np.ndarray): State vector [x, y, vx, vy]
        particle_id (int): Unique identifier for the particle
    """

    # Class variable to track particle IDs
    _next_id = 0

    def __init__(self,
                 x: float,
                 y: float,
                 vx: float,
                 vy: float,
                 mass: Optional[float] = None,
                 charge: Optional[float] = None):
        """
        Initialize a particle with given position and velocity.

        Args:
            x: Initial x-coordinate
            y: Initial y-coordinate
            vx: Initial x-velocity
            vy: Initial y-velocity
            mass: Particle mass (uses constant if not specified)
            charge: Particle charge (uses constant if not specified)
        """
        # Assign unique ID to this particle
        self.particle_id = Particle._next_id
        Particle._next_id += 1

        # Set physical properties
        self.mass = mass if mass is not None else const.MASS
        self.charge = charge if charge is not None else const.CHARGE

        # Initialize state vector [x, y, vx, vy]
        # We use a numpy array for efficient vector operations
        self.state = np.array([x, y, vx, vy], dtype=np.float64)

        # Store initial state for energy calculations and analysis
        self.initial_state = self.state.copy()

        # Track collision history for debugging
        self.collision_count = 0
        self.last_collision_time = -1.0

    @property
    def position(self) -> np.ndarray:
        """
        Get the position vector [x, y] of the particle.

        Returns:
            np.ndarray: Position vector
        """
        return self.state[0:2]

    @position.setter
    def position(self, pos: np.ndarray):
        """
        Set the position of the particle.

        Args:
            pos: New position vector [x, y]
        """
        self.state[0:2] = pos

    @property
    def velocity(self) -> np.ndarray:
        """
        Get the velocity vector [vx, vy] of the particle.

        Returns:
            np.ndarray: Velocity vector
        """
        return self.state[2:4]

    @velocity.setter
    def velocity(self, vel: np.ndarray):
        """
        Set the velocity of the particle.

        Args:
            vel: New velocity vector [vx, vy]
        """
        self.state[2:4] = vel

    @property
    def x(self) -> float:
        """Get x-coordinate."""
        return self.state[0]

    @property
    def y(self) -> float:
        """Get y-coordinate."""
        return self.state[1]

    @property
    def vx(self) -> float:
        """Get x-velocity."""
        return self.state[2]

    @property
    def vy(self) -> float:
        """Get y-velocity."""
        return self.state[3]

    def kinetic_energy(self) -> float:
        """
        Calculate the kinetic energy of the particle.

        KE = (1/2) * m * (vx^2 + vy^2)

        Returns:
            float: Kinetic energy
        """
        v_squared = np.dot(self.velocity, self.velocity)
        return 0.5 * self.mass * v_squared

    def potential_energy_gravity(self) -> float:
        """
        Calculate gravitational potential energy.

        PE_gravity = -m * g * y
        Note: We use -g because gravity constant is negative

        Returns:
            float: Gravitational potential energy
        """
        return -self.mass * const.GRAVITY * self.y

    def distance_to(self, other: 'Particle') -> float:
        """
        Calculate distance to another particle.

        Args:
            other: Another Particle object

        Returns:
            float: Distance between particles
        """
        displacement = self.position - other.position
        return np.linalg.norm(displacement)

    def displacement_to(self, other: 'Particle') -> np.ndarray:
        """
        Calculate displacement vector to another particle.

        The displacement points FROM other TO self.

        Args:
            other: Another Particle object

        Returns:
            np.ndarray: Displacement vector [dx, dy]
        """
        return self.position - other.position

    def update_state(self, new_state: np.ndarray):
        """
        Update the particle state with a new state vector.

        Args:
            new_state: New state vector [x, y, vx, vy]
        """
        # Validate state vector size
        if len(new_state) != 4:
            raise ValueError(f"State vector must have 4 components, got {len(new_state)}")

        # Update the state - ensure it's a copy to avoid reference issues
        self.state = np.array(new_state, dtype=np.float64)

    def copy(self) -> 'Particle':
        """
        Create a deep copy of the particle.

        Returns:
            Particle: A new Particle object with same properties
        """
        new_particle = Particle(
            self.x, self.y, self.vx, self.vy,
            mass=self.mass, charge=self.charge
        )
        new_particle.collision_count = self.collision_count
        new_particle.last_collision_time = self.last_collision_time
        new_particle.initial_state = self.initial_state.copy()
        return new_particle

    def __str__(self) -> str:
        """
        String representation of the particle.

        Returns:
            str: Human-readable particle description
        """
        return (f"Particle {self.particle_id}: "
                f"pos=({self.x:.2f}, {self.y:.2f}), "
                f"vel=({self.vx:.2f}, {self.vy:.2f}), "
                f"m={self.mass}, q={self.charge}")

    def __repr__(self) -> str:
        """
        Technical representation of the particle.

        Returns:
            str: Constructor-style representation
        """
        return (f"Particle(x={self.x}, y={self.y}, "
                f"vx={self.vx}, vy={self.vy}, "
                f"mass={self.mass}, charge={self.charge})")
