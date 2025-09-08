#!/usr/bin/env python3
"""
fuehre_alle_tests.py - Haupt-Testrunner für alle Unit-Tests

Dieses Skript entdeckt und führt alle Unit-Tests in der Test-Suite aus
und bietet umfassende Coverage-Berichte und Validierung.
"""

import unittest
import sys
import os
import time
import argparse
from io import StringIO

# Füge übergeordnetes Verzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fuehre_tests_aus(verbositaet=2, muster='test_*.py', schnellfehler=False):
    """
    Führe alle Unit-Tests aus und gib Ergebnisse zurück.

    Args:
        verbositaet: Test-Ausgabe-Verbosität (0=leise, 1=normal, 2=ausführlich)
        muster: Dateimuster für Test-Entdeckung
        schnellfehler: Stoppe beim ersten Fehler

    Returns:
        TestResult-Objekt
    """
    # Erstelle Test-Loader
    loader = unittest.TestLoader()

    # Suche Tests
    test_verz = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(test_verz, pattern=muster)

    # Erstelle Test-Runner
    runner = unittest.TextTestRunner(
        verbosity=verbositaet,
        failfast=schnellfehler,
        stream=sys.stdout
    )

    # Führe Tests aus
    print("=" * 70)
    print("FÜHRE UNIT-TESTS FÜR SIMULATION GELADENER TEILCHEN AUS")
    print("=" * 70)
    print(f"Test-Verzeichnis: {test_verz}")
    print(f"Muster: {muster}")
    print()

    startzeit = time.time()
    ergebnis = runner.run(suite)
    verstrichene_zeit = time.time() - startzeit

    return ergebnis, verstrichene_zeit


def fuehre_spezifisches_testmodul_aus(modulname, verbositaet=2):
    """
    Führe Tests aus einem spezifischen Modul aus.

    Args:
        modulname: Name des Testmoduls (z.B. 'test_teilchen')
        verbositaet: Test-Ausgabe-Verbosität

    Returns:
        TestResult-Objekt
    """
    loader = unittest.TestLoader()

    try:
        # Importiere das spezifische Testmodul
        testmodul = __import__(modulname)
        suite = loader.loadTestsFromModule(testmodul)

        runner = unittest.TextTestRunner(verbosity=verbositaet)
        print(f"\nFühre Tests aus {modulname} aus")
        print("-" * 50)

        ergebnis = runner.run(suite)
        return ergebnis

    except ImportError as e:
        print(f"Fehler: Konnte {modulname} nicht importieren: {e}")
        return None


def drucke_zusammenfassung(ergebnis, verstrichene_zeit):
    """
    Drucke Testergebnis-Zusammenfassung.

    Args:
        ergebnis: TestResult-Objekt
        verstrichene_zeit: Zeit für Testausführung
    """
    print("\n" + "=" * 70)
    print("TEST-ZUSAMMENFASSUNG")
    print("=" * 70)

    # Grundlegende Statistiken
    gesamt_tests = ergebnis.testsRun
    fehler = len(ergebnis.failures)
    ausfaelle = len(ergebnis.errors)
    uebersprungen = len(ergebnis.skipped) if hasattr(ergebnis, 'skipped') else 0
    erfolge = gesamt_tests - fehler - ausfaelle - uebersprungen

    print(f"Tests ausgeführt:     {gesamt_tests}")
    print(f"Erfolge:              {erfolge}")
    print(f"Fehler:               {fehler}")
    print(f"Ausfälle:             {ausfaelle}")
    print(f"Übersprungen:         {uebersprungen}")
    print(f"Verstrichene Zeit:    {verstrichene_zeit:.2f} Sekunden")

    # Erfolgsrate
    if gesamt_tests > 0:
        erfolgsrate = (erfolge / gesamt_tests) * 100
        print(f"Erfolgsrate:          {erfolgsrate:.1f}%")

    print("=" * 70)

    # Detaillierte Fehlerinformationen
    if fehler:
        print("\nFEHLER:")
        print("-" * 40)
        for test, traceback in ergebnis.failures:
            print(f"\n{test}:")
            print(traceback)

    if ausfaelle:
        print("\nAUSFÄLLE:")
        print("-" * 40)
        for test, traceback in ergebnis.errors:
            print(f"\n{test}:")
            print(traceback)

    # Finaler Status
    print("\n" + "=" * 70)
    if ergebnis.wasSuccessful():
        print("✅ ALLE TESTS BESTANDEN!")
    else:
        print("❌ EINIGE TESTS FEHLGESCHLAGEN")
    print("=" * 70)


def fuehre_coverage_analyse_aus():
    """
    Führe Tests mit Code-Coverage-Analyse aus.

    Benötigt: pip install coverage
    """
    try:
        import coverage

        print("Führe Tests mit Coverage-Analyse aus...")
        print("-" * 50)

        # Starte Coverage
        cov = coverage.Coverage(source=['src'])
        cov.start()

        # Führe Tests aus
        ergebnis, verstrichene_zeit = fuehre_tests_aus(verbositaet=1)

        # Stoppe Coverage
        cov.stop()
        cov.save()

        # Drucke Coverage-Bericht
        print("\n" + "=" * 70)
        print("COVERAGE-BERICHT")
        print("=" * 70)

        # Erstelle String-Puffer für Bericht
        puffer = StringIO()
        cov.report(file=puffer)
        print(puffer.getvalue())

        # Generiere HTML-Bericht
        print("\nGeneriere HTML-Coverage-Bericht...")
        cov.html_report(directory='htmlcov')
        print("HTML-Bericht gespeichert in: htmlcov/index.html")

        return ergebnis

    except ImportError:
        print("Coverage-Modul nicht installiert.")
        print("Installiere mit: pip install coverage")
        print("Führe Tests ohne Coverage aus...")
        ergebnis, verstrichene_zeit = fuehre_tests_aus()
        return ergebnis


def haupt():
    """Haupteinstiegspunkt für Test-Runner."""
    parser = argparse.ArgumentParser(
        description='Führe Unit-Tests für Simulation geladener Teilchen aus'
    )

    parser.add_argument(
        '-v', '--verbositaet',
        type=int,
        choices=[0, 1, 2],
        default=2,
        help='Test-Ausgabe-Verbosität (0=leise, 1=normal, 2=ausführlich)'
    )

    parser.add_argument(
        '-p', '--muster',
        default='test_*.py',
        help='Dateimuster für Test-Entdeckung'
    )

    parser.add_argument(
        '-f', '--schnellfehler',
        action='store_true',
        help='Stoppe beim ersten Testfehler'
    )

    parser.add_argument(
        '-m', '--modul',
        help='Führe spezifisches Testmodul aus (z.B. test_teilchen)'
    )

    parser.add_argument(
        '-c', '--coverage',
        action='store_true',
        help='Führe mit Code-Coverage-Analyse aus'
    )

    parser.add_argument(
        '--nur-kritisch',
        action='store_true',
        help='Führe nur kritische Tests aus (Kräfte, Integrator, Box)'
    )

    args = parser.parse_args()

    # Führe Coverage-Analyse aus falls angefordert
    if args.coverage:
        ergebnis = fuehre_coverage_analyse_aus()
        return 0 if ergebnis.wasSuccessful() else 1

    # Führe spezifisches Modul aus falls angefordert
    if args.modul:
        ergebnis = fuehre_spezifisches_testmodul_aus(args.modul, args.verbositaet)
        return 0 if ergebnis and ergebnis.wasSuccessful() else 1

    # Führe nur kritische Tests aus falls angefordert
    if args.nur_kritisch:
        print("Führe nur kritische Tests aus...")
        kritische_module = [
            'test_kraefte',
            'test_integrator',
            'test_box',
            'test_regularisierung'
        ]

        alle_bestanden = True
        for modul in kritische_module:
            ergebnis = fuehre_spezifisches_testmodul_aus(modul, args.verbositaet)
            if ergebnis and not ergebnis.wasSuccessful():
                alle_bestanden = False

        return 0 if alle_bestanden else 1

    # Führe alle Tests aus
    ergebnis, verstrichene_zeit = fuehre_tests_aus(
        verbositaet=args.verbositaet,
        muster=args.muster,
        schnellfehler=args.schnellfehler
    )

    # Drucke Zusammenfassung
    drucke_zusammenfassung(ergebnis, verstrichene_zeit)

    # Gib Exit-Code zurück
    return 0 if ergebnis.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(haupt())
