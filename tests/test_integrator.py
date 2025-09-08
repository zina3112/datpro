"""
test_integrator.py - Unit-Tests für RK4-Integration

Testet die Runge-Kutta 4. Ordnung Integrationsmethode auf Genauigkeit und Stabilität.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.teilchen import Teilchen
from src.integrator import (
    rk4_schritt_einzeln,
    rk4_schritt_system,
    RK4Integrator,
    zustandsableitung
)
import src.konstanten as konst


class TestIntegrator(unittest.TestCase):
    """Test-Suite für RK4-Integration."""

    def setUp(self):
        """Setze Test-Teilchen und Integrator auf."""
        # Einfaches Zwei-Teilchen-System
        self.teilchen = [
            Teilchen(x=10.0, y=50.0, vx=5.0, vy=0.0),
            Teilchen(x=90.0, y=50.0, vx=-5.0, vy=0.0)
        ]

        self.integrator = RK4Integrator(dt=0.001)

    def test_zustandsableitung_struktur(self):
        """Teste, dass Zustandsableitung korrekte Struktur hat."""
        ableitungen = zustandsableitung(self.teilchen)

        # Sollte Liste von Ableitungen für jedes Teilchen zurückgeben
        self.assertEqual(len(ableitungen), len(self.teilchen))

        for ableit in ableitungen:
            # Jede Ableitung sollte 4D sein: [vx, vy, ax, ay]
            self.assertEqual(len(ableit), 4)
            self.assertIsInstance(ableit, np.ndarray)

    def test_zustandsableitung_geschwindigkeitskomponenten(self):
        """Teste, dass Geschwindigkeitskomponenten korrekt in Ableitung platziert werden."""
        ableitungen = zustandsableitung(self.teilchen)

        for i, (teilchen, ableit) in enumerate(zip(self.teilchen, ableitungen)):
            # Erste zwei Komponenten sollten Geschwindigkeit sein
            np.testing.assert_array_almost_equal(ableit[0:2], teilchen.geschwindigkeit)

    def test_rk4_einzelnes_teilchen_freier_fall(self):
        """Teste RK4-Integration für ein Teilchen im freien Fall."""
        # Einzelnes Teilchen fallend unter Gravitation
        teilchen = Teilchen(x=50.0, y=80.0, vx=0.0, vy=0.0, masse=1.0, ladung=0.0)
        teilchen_liste = [teilchen]
        dt = 0.01

        # Berechne einen RK4-Schritt
        inkrement = rk4_schritt_einzeln(teilchen, teilchen_liste, 0, dt)

        # Nach kleiner Zeit dt sollte Teilchen sich leicht nach unten bewegen
        # y_neu ≈ y + vy*dt + 0.5*g*dt^2
        # Mit vy=0 anfänglich: Δy ≈ 0.5*g*dt^2
        erwartetes_dy = 0.5 * konst.GRAVITATION * dt**2

        # Prüfe y-Verschiebung (inkrement[1] ist Δy)
        self.assertLess(inkrement[1], 0)  # Sollte negativ sein (fallend)
        self.assertAlmostEqual(inkrement[1], erwartetes_dy, places=4)

    def test_rk4_system_erhaltung(self):
        """Teste, dass RK4 bestimmte Erhaltungsgesetze bewahrt."""
        # Zwei Teilchen ohne Ladung (nur Gravitation)
        teilchen = [
            Teilchen(x=30.0, y=50.0, vx=10.0, vy=0.0, ladung=0.0),
            Teilchen(x=70.0, y=50.0, vx=-10.0, vy=0.0, ladung=0.0)
        ]

        # Berechne Schwerpunkt-Geschwindigkeit
        gesamt_impuls_x = sum(p.masse * p.vx for p in teilchen)
        gesamt_masse = sum(p.masse for p in teilchen)
        schwerpunkt_vx = gesamt_impuls_x / gesamt_masse

        # Führe einen RK4-Schritt aus
        inkremente = rk4_schritt_system(teilchen, 0.001)

        # Wende Inkremente an
        for p, ink in zip(teilchen, inkremente):
            p.aktualisiere_zustand(p.zustand + ink)

        # Prüfe Impulserhaltung in x (keine horizontalen Kräfte)
        neuer_impuls_x = sum(p.masse * p.vx for p in teilchen)
        self.assertAlmostEqual(neuer_impuls_x, gesamt_impuls_x, places=10)

    def test_rk4_genauigkeit_harmonischer_oszillator(self):
        """Teste RK4-Genauigkeit mit einem harmonischen Oszillator-Analogon."""
        # Erstelle ein Teilchen, das an einer "Feder" befestigt ist (unter modifizierten Kräften)
        # Dies testet die Integrationsgenauigkeit für eine bekannte Lösung

        # Wir simulieren ein Teilchen mit einer Rückstellkraft
        # Für diesen Test verwenden wir ein einzelnes Teilchen und prüfen Energieerhaltung
        teilchen = Teilchen(x=55.0, y=50.0, vx=0.0, vy=10.0)
        teilchen_liste = [teilchen]

        anfangsenergie = teilchen.kinetische_energie() + teilchen.potentielle_energie_gravitation()

        # Führe mehrere Schritte aus
        dt = 0.001
        for _ in range(100):
            inkrement = rk4_schritt_einzeln(teilchen, teilchen_liste, 0, dt)
            teilchen.aktualisiere_zustand(teilchen.zustand + inkrement)

        endenergie = teilchen.kinetische_energie() + teilchen.potentielle_energie_gravitation()

        # Energie sollte sich nicht dramatisch ändern (etwas Drift ist erwartet)
        energieaenderung = abs(endenergie - anfangsenergie)
        relative_aenderung = energieaenderung / abs(anfangsenergie) if anfangsenergie != 0 else 0

        # RK4 sollte Energie innerhalb von 0.1% für diese kurze Simulation erhalten
        self.assertLess(relative_aenderung, 0.001)

    def test_rk4_zeitschritt_konsistenz(self):
        """Teste, dass kleinere Zeitschritte genauere Ergebnisse liefern."""
        # Setze identische Anfangsbedingungen auf
        teilchen_grob = [
            Teilchen(x=50.0, y=50.0, vx=10.0, vy=10.0)
        ]
        teilchen_fein = [
            Teilchen(x=50.0, y=50.0, vx=10.0, vy=10.0)
        ]

        # Integriere mit unterschiedlichen Zeitschritten
        dt_grob = 0.01
        dt_fein = 0.001
        gesamtzeit = 0.1

        # Grobe Integration
        schritte_grob = int(gesamtzeit / dt_grob)
        for _ in range(schritte_grob):
            ink = rk4_schritt_system(teilchen_grob, dt_grob)
            teilchen_grob[0].aktualisiere_zustand(teilchen_grob[0].zustand + ink[0])

        # Feine Integration
        schritte_fein = int(gesamtzeit / dt_fein)
        for _ in range(schritte_fein):
            ink = rk4_schritt_system(teilchen_fein, dt_fein)
            teilchen_fein[0].aktualisiere_zustand(teilchen_fein[0].zustand + ink[0])

        # Positionen sollten ähnlich sein, aber fein sollte genauer sein
        pos_diff = np.linalg.norm(teilchen_fein[0].position - teilchen_grob[0].position)

        # Es sollte einen Unterschied geben (grob ist weniger genau)
        self.assertGreater(pos_diff, 0.0)
        # Aber nicht zu viel für diesen einfachen Fall
        self.assertLess(pos_diff, 1.0)

    def test_rk4_integrator_klasse(self):
        """Teste die RK4Integrator-Klassen-Funktionalität."""
        integrator = RK4Integrator(dt=0.001)

        # Teste Initialisierung
        self.assertEqual(integrator.dt, 0.001)
        self.assertEqual(integrator.schrittzaehler, 0)
        self.assertEqual(integrator.gesamtzeit, 0.0)

        # Teste Schritt-Methode
        teilchen = [Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0)]
        inkremente = integrator.schritt(teilchen)

        self.assertEqual(len(inkremente), 1)
        self.assertEqual(integrator.schrittzaehler, 1)
        self.assertAlmostEqual(integrator.gesamtzeit, 0.001)

        # Teste Zurücksetzen
        integrator.zuruecksetzen()
        self.assertEqual(integrator.schrittzaehler, 0)
        self.assertEqual(integrator.gesamtzeit, 0.0)

    def test_rk4_mit_starker_abstossung(self):
        """Teste RK4-Stabilität mit starker Coulomb-Abstoßung."""
        # Zwei hochgeladene Teilchen nah beieinander
        teilchen = [
            Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0),
            Teilchen(x=51.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)
        ]

        # Sollte starke Abstoßung ohne NaN oder Unendlich behandeln
        inkremente = rk4_schritt_system(teilchen, 0.001)

        for ink in inkremente:
            self.assertTrue(np.isfinite(ink).all())
            # Teilchen sollten sich auseinander bewegen
            self.assertNotEqual(np.linalg.norm(ink), 0.0)

    def test_rk4_null_zeitschritt(self):
        """Teste, dass Null-Zeitschritt Null-Inkrement zurückgibt."""
        teilchen = Teilchen(x=50.0, y=50.0, vx=10.0, vy=10.0)
        teilchen_liste = [teilchen]

        inkrement = rk4_schritt_einzeln(teilchen, teilchen_liste, 0, dt=0.0)
        np.testing.assert_array_almost_equal(inkrement, np.zeros(4))

    def test_rk4_koeffizienten(self):
        """Teste, dass RK4 korrekte Koeffizientengewichte verwendet."""
        # Dies wird implizit durch Genauigkeit getestet, aber wir können die Struktur verifizieren
        # indem wir prüfen, dass die Methode 4. Ordnung genau ist

        # Für eine lineare ODE sollte RK4 bis auf Maschinengenauigkeit exakt sein
        # Teste mit Teilchen unter konstanter Kraft (nur Gravitation, kein Coulomb)
        teilchen = Teilchen(x=50.0, y=100.0, vx=0.0, vy=0.0, ladung=0.0)
        teilchen_liste = [teilchen]

        dt = 0.01
        inkrement = rk4_schritt_einzeln(teilchen, teilchen_liste, 0, dt)

        # Für konstante Beschleunigung, exakte Lösung ist:
        # Δy = vy*dt + 0.5*ay*dt^2
        # Δvy = ay*dt
        ay = konst.GRAVITATION  # Beschleunigung durch Gravitation
        erwartetes_dvy = ay * dt
        erwartetes_dy = 0.0 * dt + 0.5 * ay * dt**2

        # RK4 sollte dies für konstante Beschleunigung exakt richtig bekommen
        self.assertAlmostEqual(inkrement[3], erwartetes_dvy, places=10)
        # Y-Verschiebung ist komplexer aufgrund von RK4s Mittelung
        # sollte aber sehr nah an der analytischen Lösung sein
        self.assertAlmostEqual(inkrement[1], erwartetes_dy, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)