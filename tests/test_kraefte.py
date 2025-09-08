"""
test_kraefte.py - Unit-Tests für Kraftberechnungen

Testet Gravitations- und Coulomb-Kräfte, Energieberechnungen und Singularitätsbehandlung.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.teilchen import Teilchen
from src.kraefte import (
    berechne_gravitationskraft,
    berechne_coulombkraft_zwischen,
    berechne_gesamte_elektrostatische_kraft,
    berechne_gesamtkraft,
    berechne_beschleunigung,
    berechne_potentielle_energie_coulomb,
    berechne_system_kraefte_symmetrisch
)
import src.konstanten as konst


class TestKraefte(unittest.TestCase):
    """Test-Suite für Kraftberechnungen."""

    def setUp(self):
        """Setze Test-Teilchen auf."""
        # Erstelle Teilchen an bekannten Positionen
        self.teilchen1 = Teilchen(x=0.0, y=0.0, vx=0.0, vy=0.0)
        self.teilchen2 = Teilchen(x=3.0, y=4.0, vx=0.0, vy=0.0)  # Abstand = 5
        self.teilchen3 = Teilchen(x=10.0, y=0.0, vx=0.0, vy=0.0)  # Abstand = 10 von p1

        self.teilchen = [self.teilchen1, self.teilchen2, self.teilchen3]

    def test_gravitationskraft(self):
        """Teste Gravitationskraft-Berechnung."""
        kraft = berechne_gravitationskraft(self.teilchen1)

        # Kraft sollte [0, m*g] sein, wobei g = -10
        erwartete_kraft = np.array([0.0, konst.MASSE * konst.GRAVITATION])
        np.testing.assert_array_almost_equal(kraft, erwartete_kraft)

        # Teste mit unterschiedlicher Masse
        teilchen_schwer = Teilchen(0, 0, 0, 0, masse=2.0)
        kraft_schwer = berechne_gravitationskraft(teilchen_schwer)
        erwartete_schwer = np.array([0.0, 2.0 * konst.GRAVITATION])
        np.testing.assert_array_almost_equal(kraft_schwer, erwartete_schwer)

    def test_coulombkraft_zwischen_teilchen(self):
        """Teste Coulomb-Kraft zwischen zwei Teilchen."""
        # Teilchen bei (0,0) und (3,4), Abstand = 5
        kraft = berechne_coulombkraft_zwischen(self.teilchen1, self.teilchen2)

        # Kraftbetrag = q1*q2/r^2 = 50*50/25 = 100
        erwarteter_betrag = (konst.LADUNG * konst.LADUNG) / 25.0
        tatsaechlicher_betrag = np.linalg.norm(kraft)
        self.assertAlmostEqual(tatsaechlicher_betrag, erwarteter_betrag, places=5)

        # Kraftrichtung: von p2 zu p1, sollte also in (-3, -4) Richtung zeigen
        # Normalisiert: (-3/5, -4/5) * Betrag
        erwartete_richtung = np.array([-3.0/5.0, -4.0/5.0])
        tatsaechliche_richtung = kraft / tatsaechlicher_betrag
        np.testing.assert_array_almost_equal(tatsaechliche_richtung, erwartete_richtung)

    def test_coulombkraft_symmetrie(self):
        """Teste Newtons drittes Gesetz für Coulomb-Kräfte."""
        kraft_1_auf_2 = berechne_coulombkraft_zwischen(self.teilchen2, self.teilchen1)
        kraft_2_auf_1 = berechne_coulombkraft_zwischen(self.teilchen1, self.teilchen2)

        # Kräfte sollten gleich und entgegengesetzt sein
        np.testing.assert_array_almost_equal(kraft_1_auf_2, -kraft_2_auf_1)

    def test_coulombkraft_singularitaetsbehandlung(self):
        """Teste, dass nahe Teilchen keine numerische Explosion verursachen."""
        # Erstelle zwei Teilchen sehr nah beieinander
        p1 = Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0)
        p2 = Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0)  # Exakt überlappend

        # Sollte Null-Kraft für überlappende Teilchen zurückgeben
        kraft = berechne_coulombkraft_zwischen(p1, p2)
        np.testing.assert_array_almost_equal(kraft, np.array([0.0, 0.0]))

        # Teste mit sehr kleiner Trennung
        p2.position = np.array([50.0 + 1e-8, 50.0])
        kraft = berechne_coulombkraft_zwischen(p1, p2)

        # Kraft sollte groß aber endlich sein
        self.assertTrue(np.isfinite(kraft).all())
        self.assertLess(np.linalg.norm(kraft), 1e15)  # Eine vernünftige obere Grenze

    def test_gesamte_elektrostatische_kraft(self):
        """Teste gesamte elektrostatische Kraft von mehreren Teilchen."""
        # Berechne Kraft auf Teilchen 1 von Teilchen 2 und 3
        gesamtkraft = berechne_gesamte_elektrostatische_kraft(0, self.teilchen)

        # Kraft von p2 bei (3,4): abstoßend in Richtung (-3,-4)/5
        kraft_von_2 = berechne_coulombkraft_zwischen(self.teilchen1, self.teilchen2)

        # Kraft von p3 bei (10,0): abstoßend in Richtung (-10,0)/10 = (-1,0)
        kraft_von_3 = berechne_coulombkraft_zwischen(self.teilchen1, self.teilchen3)

        erwartete_gesamt = kraft_von_2 + kraft_von_3
        np.testing.assert_array_almost_equal(gesamtkraft, erwartete_gesamt)

    def test_gesamtkraft(self):
        """Teste kombinierte Gravitations- und elektrostatische Kräfte."""
        gesamtkraft = berechne_gesamtkraft(0, self.teilchen)

        gravitation = berechne_gravitationskraft(self.teilchen1)
        elektrostatisch = berechne_gesamte_elektrostatische_kraft(0, self.teilchen)
        erwartete = gravitation + elektrostatisch

        np.testing.assert_array_almost_equal(gesamtkraft, erwartete)

    def test_beschleunigungsberechnung(self):
        """Teste F=ma Beschleunigungsberechnung."""
        # Setze bekanntes Kraft-Szenario auf
        beschleunigung = berechne_beschleunigung(0, self.teilchen)

        # a = F/m
        gesamtkraft = berechne_gesamtkraft(0, self.teilchen)
        erwartete_beschl = gesamtkraft / self.teilchen1.masse

        np.testing.assert_array_almost_equal(beschleunigung, erwartete_beschl)

        # Teste mit Null-Masse-Teilchen (sollte Null-Beschleunigung zurückgeben)
        teilchen_masselos = Teilchen(0, 0, 0, 0, masse=0.0)
        teilchen_mit_masselos = [teilchen_masselos, self.teilchen2]
        beschl_masselos = berechne_beschleunigung(0, teilchen_mit_masselos)
        np.testing.assert_array_almost_equal(beschl_masselos, np.array([0.0, 0.0]))

    def test_coulomb_potentielle_energie(self):
        """Teste Coulomb-Potentialenergie-Berechnung."""
        # U = q1*q2/r für jedes Paar, summiert
        potential = berechne_potentielle_energie_coulomb(self.teilchen)

        # Abstand zwischen p1 und p2: 5
        u12 = (konst.LADUNG * konst.LADUNG) / 5.0

        # Abstand zwischen p1 und p3: 10
        u13 = (konst.LADUNG * konst.LADUNG) / 10.0

        # Abstand zwischen p2 und p3: sqrt((10-3)^2 + (0-4)^2) = sqrt(49+16) = sqrt(65)
        abst23 = np.sqrt(65)
        u23 = (konst.LADUNG * konst.LADUNG) / abst23

        erwartete_gesamt = u12 + u13 + u23
        self.assertAlmostEqual(potential, erwartete_gesamt, places=5)

    def test_system_kraefte_symmetrisch(self):
        """Teste, dass symmetrische Kraftberechnung Newtons 3. Gesetz bewahrt."""
        kraefte = berechne_system_kraefte_symmetrisch(self.teilchen)

        # Prüfe, dass totale Impulsänderung null ist (Newtons 3. Gesetz)
        # Summe aller Kräfte sollte null sein (außer Gravitation)
        gesamt_kraft_x = sum(f[0] for f in kraefte)
        gesamt_kraft_y = sum(f[1] for f in kraefte)

        # X-Komponente sollte exakt null sein
        self.assertAlmostEqual(gesamt_kraft_x, 0.0, places=10)

        # Y-Komponente sollte gleich der gesamten Gravitationskraft sein
        gesamt_gravitation = len(self.teilchen) * konst.MASSE * konst.GRAVITATION
        self.assertAlmostEqual(gesamt_kraft_y, gesamt_gravitation, places=10)

    def test_kraftberechnung_mit_tatsaechlichen_parametern(self):
        """Teste Kräfte mit tatsächlichen Simulationsparametern (q=50, m=1)."""
        # Erstelle Teilchen mit tatsächlichen Ladungswerten
        p1 = Teilchen(x=20.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)
        p2 = Teilchen(x=30.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)

        # Abstand = 10, Kraft = 50*50/100 = 25
        kraft = berechne_coulombkraft_zwischen(p1, p2)
        erwarteter_betrag = 25.0
        tatsaechlicher_betrag = np.linalg.norm(kraft)
        self.assertAlmostEqual(tatsaechlicher_betrag, erwarteter_betrag, places=5)

    def test_energieerhaltung_im_isolierten_system(self):
        """Teste, dass Potentialenergie-Berechnung konsistent ist."""
        # Erstelle ein einfaches Zwei-Teilchen-System
        teilchen = [
            Teilchen(x=0.0, y=0.0, vx=5.0, vy=0.0),
            Teilchen(x=10.0, y=0.0, vx=-5.0, vy=0.0)
        ]

        # Berechne Gesamtenergie
        ke_gesamt = sum(p.kinetische_energie() for p in teilchen)
        pe_gravitation = sum(p.potentielle_energie_gravitation() for p in teilchen)
        pe_coulomb = berechne_potentielle_energie_coulomb(teilchen)

        gesamtenergie = ke_gesamt + pe_gravitation + pe_coulomb

        # Energie sollte wohldefiniert sein (endlich)
        self.assertTrue(np.isfinite(gesamtenergie))

    def test_kraft_skalierung_mit_abstand(self):
        """Teste, dass Coulomb-Kraft als 1/r^2 skaliert."""
        p1 = Teilchen(x=0.0, y=0.0, vx=0.0, vy=0.0)

        abstaende = [1.0, 2.0, 4.0, 8.0]
        kraefte = []

        for d in abstaende:
            p2 = Teilchen(x=d, y=0.0, vx=0.0, vy=0.0)
            kraft = berechne_coulombkraft_zwischen(p1, p2)
            kraefte.append(np.linalg.norm(kraft))

        # Prüfe 1/r^2 Skalierung
        for i in range(1, len(abstaende)):
            verhaeltnis = abstaende[i] / abstaende[i-1]
            erwartetes_kraft_verhaeltnis = 1.0 / (verhaeltnis ** 2)
            tatsaechliches_kraft_verhaeltnis = kraefte[i] / kraefte[i-1]
            self.assertAlmostEqual(tatsaechliches_kraft_verhaeltnis, erwartetes_kraft_verhaeltnis, places=5)


if __name__ == '__main__':
    unittest.main(verbosity=2)