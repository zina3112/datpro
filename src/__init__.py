"""
Paket-Initialisierungsdatei für die Simulation geladener Teilchen

Macht src zu einem Python-Paket und ermöglicht den Import
der verschiedenen Module und Klassen.
"""

# Versionsinformation
__version__ = '1.0.0'
__author__ = 'DATPRO Simulationsteam'

# Importiere Hauptklassen für einfachen Zugriff
from .teilchen import Teilchen
from .box import Box
from .simulation import Simulation
from .visualisierung import Visualisierer
from .datenverwalter import Datenverwalter
from .integrator import RK4Integrator

# Exportiere öffentliche API
__all__ = [
    'Teilchen',
    'Box',
    'Simulation',
    'Visualisierer',
    'Datenverwalter',
    'RK4Integrator',
]
