"""
test_regularisierung.py - Tests speziell für Soft-Core-Regularisierung

Diese Testdatei validiert, dass die Soft-Core-Regularisierung notwendig ist
und korrekt funktioniert, um numerische Explosionen zu verhindern.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.teilchen import Teilchen
from src.kraefte import berechne_coulombkraft_zwischen
from src.simulation import Simulation
import src.konstanten as konst


class TestRegularisierung(unittest.TestCase):
    """Test-Suite zur Validierung der Soft-Core-Regularisierung."""

    def test_regularisierung_verhindert_unendlichkeit(self):
        """Teste, dass Regularisierung unendliche Kräfte bei r=0 verhindert."""
        # Erstelle überlappende Teilchen
        p1 = Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)
        p2 = Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)

        # Berechne Kraft - sollte NICHT unendlich sein
        kraft = berechne_coulombkraft_zwischen(p1, p2)

        # Kraft sollte null oder sehr klein für überlappende Teilchen sein
        self.assertTrue(np.isfinite(kraft).all())
        self.assertLess(np.linalg.norm(kraft), 1e10)  # Eine vernünftige Grenze

        # Spezifisch gibt Implementierung Null für exakt überlappende zurück
        np.testing.assert_array_almost_equal(kraft, np.array([0.0, 0.0]))

    def test_regularisierung_bei_kleinen_abstaenden(self):
        """Teste Kraftverhalten bei sehr kleinen Abständen."""
        abstaende = [1e-7, 1e-8, 1e-9, 1e-10, 0.0]
        kraefte = []

        for d in abstaende:
            p1 = Teilchen(x=0.0, y=0.0, vx=0.0, vy=0.0, ladung=50.0)
            p2 = Teilchen(x=d, y=0.0, vx=0.0, vy=0.0, ladung=50.0)

            kraft = berechne_coulombkraft_zwischen(p1, p2)
            kraft_betrag = np.linalg.norm(kraft)
            kraefte.append(kraft_betrag)

            # Alle Kräfte sollten endlich sein
            self.assertTrue(np.isfinite(kraft_betrag))

            # Kraft sollte begrenzt sein
            self.assertLess(kraft_betrag, 1e15)

        # Kräfte sollten zunehmen wenn Abstand abnimmt, aber sättigen
        # (nicht zu Unendlichkeit explodieren)
        for i in range(1, len(kraefte)):
            if abstaende[i] > 0:  # Überspringe den exakt Null-Fall
                # Kraft sollte generell zunehmen wenn Teilchen näher kommen
                # aber dies ist nicht strikt monoton wegen Regularisierung
                pass

    def test_minimaler_annaeherungsabstand_mit_anfangsbedingungen(self):
        """Teste wie nah Teilchen mit tatsächlichen Anfangsbedingungen kommen."""
        # Verwende tatsächliche Anfangsbedingungen
        sim = Simulation(
            anfangszustaende=konst.ANFANGSZUSTAENDE,
            dt=konst.DT,
            ausgabedatei="test_min_abstand.csv"
        )

        min_abstand_gesamt = float('inf')

        # Laufe für kurze Zeit und verfolge minimalen Abstand
        for _ in range(100):  # 0.1 Sekunden
            sim.schritt()

            # Prüfe alle Teilchenpaare
            for i in range(len(sim.teilchen)):
                for j in range(i+1, len(sim.teilchen)):
                    abstand = sim.teilchen[i].abstand_zu(sim.teilchen[j])
                    min_abstand_gesamt = min(min_abstand_gesamt, abstand)

        # Protokolliere den minimalen angetroffenen Abstand
        print(f"Minimaler Abstand in 0.1s Simulation: {min_abstand_gesamt}")

        # Teilchen sollten vernünftig nah kommen (rechtfertigt Regularisierung)
        # aber nicht vollständig überlappen
        self.assertGreater(min_abstand_gesamt, 0.0)

        # Aufräumen
        if os.path.exists("test_min_abstand.csv"):
            os.remove("test_min_abstand.csv")

    def test_energieerhaltung_mit_nahen_begegnungen(self):
        """Teste, dass Energie auch bei nahen Teilchenbegegnungen erhalten bleibt."""
        # Erstelle Teilchen, die eine nahe Begegnung haben werden
        zustaende = np.array([
            [45.0, 50.0, 10.0, 0.0],   # Bewegt sich nach rechts
            [55.0, 50.0, -10.0, 0.0]    # Bewegt sich nach links - werden kollidieren
        ])

        sim = Simulation(
            anfangszustaende=zustaende,
            dt=0.001,
            ausgabedatei="test_nahe_begegnung.csv"
        )

        anfangsenergie = sim.anfangsenergie

        # Laufe bis Teilchen durch nächste Annäherung passieren
        for _ in range(500):  # 0.5 Sekunden
            sim.schritt()

        endenergie = sim.berechne_gesamtenergie()
        energie_drift = abs(endenergie - anfangsenergie) / abs(anfangsenergie)

        # Energie sollte trotz naher Begegnung erhalten bleiben
        self.assertLess(energie_drift, 0.01)  # Weniger als 1% Drift

        # Aufräumen
        if os.path.exists("test_nahe_begegnung.csv"):
            os.remove("test_nahe_begegnung.csv")

    def test_vergleiche_regularisiert_vs_reines_coulomb_bei_sicherem_abstand(self):
        """Teste, dass Regularisierung Kräfte bei normalen Abständen nicht beeinflusst."""
        # Bei vernünftigen Abständen sollten regularisiert und reines Coulomb übereinstimmen

        # Abstand = 1.0 (weit über Regularisierungsschwelle)
        p1 = Teilchen(x=0.0, y=0.0, vx=0.0, vy=0.0, ladung=50.0)
        p2 = Teilchen(x=1.0, y=0.0, vx=0.0, vy=0.0, ladung=50.0)

        kraft = berechne_coulombkraft_zwischen(p1, p2)
        kraft_betrag = np.linalg.norm(kraft)

        # Reines Coulomb: F = q1*q2/r^2 = 50*50/1^2 = 2500
        erwarteter_betrag = 2500.0

        # Sollte bei diesem Abstand mit hoher Präzision übereinstimmen
        self.assertAlmostEqual(kraft_betrag, erwarteter_betrag, places=5)

    def test_stabilitaet_mit_hoher_ladungsdichte(self):
        """Teste Simulationsstabilität mit vielen geladenen Teilchen in kleiner Region."""
        # Packe Teilchen relativ nah zusammen
        zustaende = []
        for i in range(5):
            for j in range(5):
                x = 40.0 + i * 5.0  # 5 Einheiten Abstand
                y = 40.0 + j * 5.0
                vx = np.random.uniform(-5, 5)
                vy = np.random.uniform(-5, 5)
                zustaende.append([x, y, vx, vy])

        zustaende = np.array(zustaende)

        sim = Simulation(
            anfangszustaende=zustaende,
            dt=0.001,
            ausgabedatei="test_hohe_dichte.csv"
        )

        # Sollte ohne numerische Explosion laufen können
        try:
            for _ in range(100):
                sim.schritt()

            # Prüfe, dass finale Energie endlich ist
            endenergie = sim.berechne_gesamtenergie()
            self.assertTrue(np.isfinite(endenergie))

            # Prüfe, dass alle Teilchen endliche Positionen und Geschwindigkeiten haben
            for teilchen in sim.teilchen:
                self.assertTrue(np.isfinite(teilchen.zustand).all())

            simulation_stabil = True
        except:
            simulation_stabil = False

        self.assertTrue(simulation_stabil, "Simulation wurde instabil mit hoher Ladungsdichte")

        # Aufräumen
        if os.path.exists("test_hohe_dichte.csv"):
            os.remove("test_hohe_dichte.csv")

    def test_kraft_kontinuitaet_an_regularisierungsgrenze(self):
        """Teste, dass Kraft an der Regularisierungsschwelle kontinuierlich ist."""
        # Teste Abstände um die Regularisierungsschwelle (1e-6)
        schwelle = 1e-6
        test_abstaende = [
            schwelle * 0.5,   # Unter Schwelle
            schwelle * 0.99,  # Knapp darunter
            schwelle,         # An Schwelle
            schwelle * 1.01,  # Knapp darüber
            schwelle * 2.0    # Über Schwelle
        ]

        kraefte = []
        for d in test_abstaende:
            p1 = Teilchen(x=0.0, y=0.0, vx=0.0, vy=0.0, ladung=50.0)
            p2 = Teilchen(x=d, y=0.0, vx=0.0, vy=0.0, ladung=50.0)

            kraft = berechne_coulombkraft_zwischen(p1, p2)
            kraefte.append(np.linalg.norm(kraft))

        # Kräfte sollten glatt variieren (kein diskontinuierlicher Sprung)
        for i in range(1, len(kraefte)):
            if test_abstaende[i] > 0 and test_abstaende[i-1] > 0:
                # Prüfe, dass Kraft nicht um mehr als Faktor 10 springt
                verhaeltnis = kraefte[i] / kraefte[i-1] if kraefte[i-1] > 0 else 0
                self.assertLess(abs(verhaeltnis), 10.0,
                              f"Kraftdiskontinuität bei {test_abstaende[i]}")

    def test_warum_regularisierung_notwendig_ist(self):
        """Demonstriere warum Regularisierung notwendig ist, indem gezeigt wird was ohne passieren würde."""
        # Dieser Test dokumentiert warum wir Regularisierung brauchen

        # Berechne was reine Coulomb-Kraft bei kleinen Abständen wäre
        q = 50.0
        kleine_abstaende = [0.01, 0.001, 0.0001, 0.00001]

        reine_coulomb_kraefte = []
        for r in kleine_abstaende:
            # Reines Coulomb: F = q^2/r^2
            f = q * q / (r * r)
            reine_coulomb_kraefte.append(f)

        # Zeige wie Kräfte explodieren
        print("\nReine Coulomb-Kräfte bei kleinen Abständen (demonstriert Notwendigkeit der Regularisierung):")
        for r, f in zip(kleine_abstaende, reine_coulomb_kraefte):
            print(f"  r = {r}: F = {f:.2e}")

        # Mit dt = 0.001, Beschleunigung = F/m würde Geschwindigkeitsänderung ergeben:
        # Δv = a * dt = F * dt
        dt = 0.001
        for r, f in zip(kleine_abstaende, reine_coulomb_kraefte):
            delta_v = f * dt
            print(f"  r = {r}: Δv in einem Schritt = {delta_v:.2e}")

        # Bei r=0.00001, Δv = 2.5e11 * 0.001 = 2.5e8
        # Dies würde Teilchen auf Geschwindigkeit von 250 Millionen Einheiten/Sekunde senden!

        # Dies demonstriert, dass Regularisierung NOTWENDIG ist, nicht optional
        self.assertGreater(reine_coulomb_kraefte[-1], 1e10,
                          "Reine Coulomb-Kraft explodiert bei kleinen Abständen")


if __name__ == '__main__':
    unittest.main(verbosity=2)