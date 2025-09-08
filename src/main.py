#!/usr/bin/env python3
"""
Hauptausführungsskript für die Simulation geladener Teilchen

Führt die vollständige Simulation geladener Teilchen mit den
spezifizierten Anfangsbedingungen durch und generiert alle erforderlichen Ausgaben.

Verwendung:
    python main.py [optionen]
"""

from src.simulation import Simulation
from src.visualisierung import Visualisierer
import src.konstanten as konst
import sys
import os
import argparse
import numpy as np
import time

# Füge übergeordnetes Verzeichnis zum Pfad hinzu damit src als Paket importiert werden kann
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def parse_argumente():
    """
    Parse Kommandozeilenargumente.

    Returns:
        argparse.Namespace: Geparste Argumente
    """
    parser = argparse.ArgumentParser(
        description='Führe Simulation geladener Teilchen in 2D-Box aus'
    )

    parser.add_argument(
        '--dt', type=float, default=konst.DT,
        help=f'Integrationszeitschritt (Standard: {konst.DT})'
    )

    parser.add_argument(
        '--zeit', type=float, default=konst.SIMULATIONSZEIT,
        help=f'Simulationszeit in Sekunden (Standard: {konst.SIMULATIONSZEIT})'
    )

    parser.add_argument(
        '--ausgabe', type=str, default=None,
        help='Ausgabedateipfad (Standard: automatisch generiert)'
    )

    parser.add_argument(
        '--keine-plots', action='store_true',
        help='Überspringe Generierung von Plots'
    )

    parser.add_argument(
        '--test', action='store_true',
        help='Führe mit Testkonfiguration aus (kürzere Simulation)'
    )

    parser.add_argument(
        '--fortschritt', type=int, default=1000,
        help='Fortschrittsaktualisierungsintervall in Schritten (Standard: 1000)'
    )

    return parser.parse_args()


def drucke_header():
    """Drucke Simulationsheader-Informationen."""
    print("=" * 70)
    print("SIMULATION GELADENER TEILCHEN IN 2D-BOX")
    print("=" * 70)
    print()
    print("Simulationsparameter:")
    print(f"  - Anzahl der Teilchen: {konst.N_TEILCHEN}")
    print(f"  - Teilchenmasse: {konst.MASSE}")
    print(f"  - Teilchenladung: {konst.LADUNG}")
    print(f"  - Gravitation: {konst.GRAVITATION}")
    print(f"  - Box-Dimensionen: [{konst.BOX_MIN_X}, {konst.BOX_MAX_X}] × "
          f"[{konst.BOX_MIN_Y}, {konst.BOX_MAX_Y}]")
    print()
    print("Anfängliche Teilchenzustände (x, y, vx, vy):")
    for i, zustand in enumerate(konst.ANFANGSZUSTAENDE):
        print(f"  Teilchen {i + 1}: {zustand}")
    print()
    print("=" * 70)
    print()


def fuehre_tests_aus():
    """
    Führe Unit-Tests aus.

    Returns:
        bool: True wenn alle Tests bestehen
    """
    print("Führe Unit-Tests aus...")
    import unittest

    # Suche und führe Tests aus
    loader = unittest.TestLoader()
    test_verz = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), 'tests')

    # Alternative falls Tests im gleichen Verzeichnis wie src sind
    if not os.path.exists(test_verz):
        test_verz = os.path.join(os.path.dirname(__file__), '..', 'tests')

    suite = loader.discover(test_verz, pattern='test_*.py')

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


def main():
    """Hauptausführungsfunktion."""
    # Parse Argumente
    args = parse_argumente()

    # Drucke Header
    drucke_header()

    # Führe Tests aus falls angefordert
    if args.test:
        print("TESTMODUS: Führe zuerst Unit-Tests aus...")
        if not fuehre_tests_aus():
            print("FEHLER: Unit-Tests fehlgeschlagen. Beende.")
            return 1
        print("\nUnit-Tests bestanden. Fahre mit Simulation fort...\n")

        # Verwende kürzere Simulation für Testmodus
        simulationszeit = 1.0
        print(
            f"TESTMODUS: Führe verkürzte Simulation aus ({simulationszeit}s)")
    else:
        simulationszeit = args.zeit

    # Validiere Zeitschritt
    if abs(args.dt) < 1e-10 and args.dt != 0:
        print(f"Warnung: Zeitschritt {
              args.dt} ist sehr klein, verwende 0.001 stattdessen")
        dt = 0.001
    else:
        dt = args.dt

    # Erstelle Ausgabedateinamen mit Zeitstempel falls nicht spezifiziert
    if args.ausgabe is None:
        zeitstempel = time.strftime("%Y%m%d_%H%M%S")
        ausgabedatei = os.path.join(
            konst.AUSGABE_VERZEICHNIS,
            f"simulation_{zeitstempel}.csv"
        )
    else:
        ausgabedatei = args.ausgabe

    # Stelle sicher dass Ausgabeverzeichnis existiert
    os.makedirs(os.path.dirname(ausgabedatei) or '.', exist_ok=True)

    print(f"Konfiguration:")
    print(f"  - Zeitschritt: {dt}")
    print(f"  - Simulationszeit: {simulationszeit}")
    print(f"  - Ausgabedatei: {ausgabedatei}")
    print(f"  - Generiere Plots: {not args.keine_plots}")
    print()

    # Erstelle und führe Simulation aus
    print("Initialisiere Simulation...")
    sim = Simulation(
        anfangszustaende=konst.ANFANGSZUSTAENDE,
        dt=dt,
        ausgabedatei=ausgabedatei
    )

    # Führe Simulation aus
    sim.laufen(
        simulationszeit=simulationszeit,
        fortschritts_intervall=args.fortschritt
    )

    # Generiere Visualisierungen
    if not args.keine_plots:
        print("\nGeneriere Visualisierungen...")

        visualisierer = Visualisierer(sim.datenverwalter)

        # Aufgabe 5: Plotte Energie vs Zeit
        print("  - Erstelle Energieerhaltungsplot...")
        visualisierer.plotte_energie_vs_zeit(speichern=True, anzeigen=False)

        # Aufgabe 6: Plotte Trajektorie des ersten Teilchens
        print("  - Erstelle Trajektorienplot für Teilchen 1...")
        visualisierer.plotte_teilchen_trajektorie(
            0, speichern=True, anzeigen=False)

        # Aufgabe 7: Plotte alle Trajektorien
        print("  - Erstelle kombinierten Trajektorienplot...")
        visualisierer.plotte_alle_trajektorien(speichern=True, anzeigen=False)

        # Zusätzliche Visualisierungen
        print("  - Erstelle vollständigen Visualisierungsbericht...")
        visualisierer.erstelle_zusammenfassungsbericht()

        print(f"\nPlots gespeichert in: {
              visualisierer.abbildungs_verzeichnis}")

    print("\n" + "=" * 70)
    print("SIMULATION ABGESCHLOSSEN")
    print("=" * 70)

    # Drucke finale Zusammenfassung
    statistiken = sim.datenverwalter.hole_statistiken()
    print("\nFinale Statistiken:")
    print(f"  - Gesamte Zeitschritte: {statistiken['anzahl_zeitschritte']}")
    print(f"  - Anfangsenergie: {statistiken['anfangsenergie']:.6f}")
    print(f"  - Endenergie: {statistiken['endenergie']:.6f}")
    print(f"  - Energiedrift: {statistiken['energie_drift']:.6e}")
    print(f"  - Relative Drift: {statistiken['relative_drift'] * 100:.4f}%")

    # Erfolgsmeldung
    print("\n✓ Alle Aufgaben erfolgreich abgeschlossen!")
    print(f"✓ Ergebnisse gespeichert in: {ausgabedatei}")
    if not args.keine_plots:
        print(f"✓ Plots gespeichert in: {
              visualisierer.abbildungs_verzeichnis}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
