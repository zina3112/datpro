"""
Physikalische und numerische Konstanten für die Simulation geladener Teilchen

Alle Parameter der Simulation sind hier zentral definiert um einfache
Anpassungen zu ermöglichen.
"""

import numpy as np

# ============================================================================
# PHYSIKALISCHE KONSTANTEN
# ============================================================================

# Eigenschaften der Teilchen
MASSE = 1.0  # Masse jedes Teilchens (dimensionslose Einheiten)
LADUNG = 50.0  # Ladung jedes Teilchens (dimensionslose Einheiten)

# Externe Kräfte
GRAVITATION = -10.0  # Gravitationsbeschleunigung in y-Richtung (negativ = nach unten)

# ============================================================================
# SIMULATIONSBOX
# ============================================================================

# Dimensionen der Box (100x100 wie in Spezifikation)
BOX_MIN_X = 0.0  # Linke Grenze
BOX_MAX_X = 100.0  # Rechte Grenze
BOX_MIN_Y = 0.0  # Untere Grenze
BOX_MAX_Y = 100.0  # Obere Grenze

# Praktische Größenvariablen
BOX_BREITE = BOX_MAX_X - BOX_MIN_X
BOX_HOEHE = BOX_MAX_Y - BOX_MIN_Y

# ============================================================================
# NUMERISCHE PARAMETER
# ============================================================================

# Zeitintegration
DT = 0.001  # Zeitschrittgröße in Sekunden
SIMULATIONSZEIT = 10.0  # Gesamte Simulationszeit in Sekunden
N_SCHRITTE = int(SIMULATIONSZEIT / DT)  # Anzahl der Zeitschritte

# Numerische Toleranzen
EPSILON = 1e-10  # Kleiner Wert für numerische Vergleiche
ENERGIE_TOLERANZ = 1e-6  # Toleranz für Energieerhaltungsprüfung

# Kollisionsbehandlung
KOLLISIONS_EPSILON = 1e-10  # Toleranz für Wandkollisionserkennung
MAX_KOLLISIONS_ITERATIONEN = 10  # Maximale Iterationen für Eckkollisionen

# ============================================================================
# ANFANGSBEDINGUNGEN
# ============================================================================

# Anfangs-Zustandsvektoren für 7 Teilchen
# Format: [x, y, vx, vy] für jedes Teilchen
ANFANGSZUSTAENDE = np.array([
    [1.0, 45.0, 10.0, 0.0],    # Teilchen 1: Bewegung nach rechts von linker Seite
    [99.0, 55.0, -10.0, 0.0],   # Teilchen 2: Bewegung nach links von rechter Seite
    [10.0, 50.0, 15.0, -15.0],  # Teilchen 3: Diagonale Bewegung
    [20.0, 30.0, -15.0, -15.0], # Teilchen 4: Diagonale Bewegung
    [80.0, 70.0, 15.0, 15.0],   # Teilchen 5: Diagonale Bewegung nach oben
    [80.0, 60.0, 15.0, 15.0],   # Teilchen 6: Gleiche Geschwindigkeit wie 5, andere Position
    [80.0, 50.0, 15.0, 15.0]    # Teilchen 7: Gleiche Geschwindigkeit wie 5&6, andere Position
])

# Anzahl der Teilchen
N_TEILCHEN = len(ANFANGSZUSTAENDE)

# ============================================================================
# AUSGABEPARAMETER
# ============================================================================

# Dateieinstellungen
AUSGABE_VERZEICHNIS = "ausgabe"
AUSGABE_DATEI = "simulationsergebnisse.csv"
PLOT_VERZEICHNIS = "plots"

# Ausgabefrequenz (1 = jeden Schritt speichern)
AUSGABE_FREQUENZ = 1

# Plot-Parameter
ABBILDUNGSGROESSE = (12, 8)  # Größe der Plots
DPI = 100  # Auflösung für gespeicherte Abbildungen
