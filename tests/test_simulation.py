"""
test_simulation.py - Unit-Tests für Hauptsimulationscontroller

Testet den vollständigen Simulationsworkflow und Energieerhaltung.
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation import Simulation
from src.teilchen import Teilchen
import src.konstanten as konst


class TestSimulation(unittest.TestCase):
    """Test-Suite für Simulations-Klasse."""

    def setUp(self):
        """Setze Test-Simulation auf."""
        # Erstelle temporäres Verzeichnis für Ausgabedateien
        self.test_verz = tempfile.mkdtemp()
        self.ausgabedatei = os.path.join(self.test_verz, "test_ausgabe.csv")

        # Einfache Testkonfiguration
        self.test_zustaende = np.array([
            [10.0, 50.0, 5.0, 0.0],
            [90.0, 50.0, -5.0, 0.0]
        ])

        self.sim = Simulation(
            anfangszustaende=self.test_zustaende,
            dt=0.001,
            ausgabedatei=self.ausgabedatei
        )

    def tearDown(self):
        """Räume temporäre Dateien auf."""
        if os.path.exists(self.test_verz):
            shutil.rmtree(self.test_verz)

    def test_initialisierung(self):
        """Teste Simulations-Initialisierung."""
        # Prüfe, dass Teilchen korrekt erstellt wurden
        self.assertEqual(len(self.sim.teilchen), 2)

        # Prüfe Anfangspositionen
        self.assertEqual(self.sim.teilchen[0].x, 10.0)
        self.assertEqual(self.sim.teilchen[1].x, 90.0)

        # Prüfe Zeitschritt
        self.assertEqual(self.sim.dt, 0.001)

        # Prüfe, dass Anfangsenergie berechnet wurde
        self.assertIsNotNone(self.sim.anfangsenergie)
        self.assertTrue(np.isfinite(self.sim.anfangsenergie))

    def test_energieberechnung(self):
        """Teste Gesamtenergieberechnung."""
        energie = self.sim.berechne_gesamtenergie()

        # Energie sollte endlich sein
        self.assertTrue(np.isfinite(energie))

        # Energie sollte Beiträge haben von:
        # - Kinetischer Energie (Teilchen bewegen sich)
        # - Gravitationspotential (Teilchen haben Höhe)
        # - Coulomb-Potential (Teilchen stoßen sich ab)

        # Grobe Prüfung: Energie sollte nicht null sein
        self.assertNotEqual(energie, 0.0)

    def test_einzelner_schritt(self):
        """Teste einzelnen Simulationsschritt."""
        anfangsenergie = self.sim.berechne_gesamtenergie()
        anfangspositionen = [p.position.copy() for p in self.sim.teilchen]

        # Führe einen Schritt aus
        erfolg = self.sim.schritt()
        self.assertTrue(erfolg)

        # Prüfe, dass Zeit vorangeschritten ist
        self.assertAlmostEqual(self.sim.aktuelle_zeit, self.sim.dt)
        self.assertEqual(self.sim.schrittzaehler, 1)

        # Prüfe, dass Teilchen sich bewegt haben
        for i, p in enumerate(self.sim.teilchen):
            # Teilchen sollten sich bewegt haben (außer im Gleichgewicht, was sie nicht sind)
            pos_aenderung = np.linalg.norm(p.position - anfangspositionen[i])
            self.assertGreater(pos_aenderung, 0.0)

        # Prüfe Energieerhaltung (sollte innerhalb Toleranz sein)
        endenergie = self.sim.berechne_gesamtenergie()
        energie_drift = abs(endenergie - anfangsenergie) / abs(anfangsenergie)
        self.assertLess(energie_drift, 0.01)  # Weniger als 1% Drift pro Schritt

    def test_mehrere_schritte(self):
        """Teste mehrere Simulationsschritte."""
        n_schritte = 10

        for _ in range(n_schritte):
            erfolg = self.sim.schritt()
            self.assertTrue(erfolg)

        self.assertEqual(self.sim.schrittzaehler, n_schritte)
        self.assertAlmostEqual(self.sim.aktuelle_zeit, n_schritte * self.sim.dt)

    def test_energieerhaltung_kurzer_lauf(self):
        """Teste Energieerhaltung über kurze Simulation."""
        # Laufe für 100 Schritte
        for _ in range(100):
            self.sim.schritt()

        endenergie = self.sim.berechne_gesamtenergie()
        energie_drift = abs(endenergie - self.sim.anfangsenergie)
        relative_drift = energie_drift / abs(self.sim.anfangsenergie)

        # Für 100 Schritte mit dt=0.001 sollte Drift sehr klein sein
        self.assertLess(relative_drift, 0.001)  # Weniger als 0.1% Gesamtdrift

    def test_wandkollisionen_treten_auf(self):
        """Teste, dass Wandkollisionen erkannt und behandelt werden."""
        # Erstelle Teilchen, das definitiv Wand treffen wird
        zustaende = np.array([[95.0, 50.0, 20.0, 0.0]])  # Schnelle Rechtsbewegung

        sim = Simulation(anfangszustaende=zustaende, dt=0.1, ausgabedatei=self.ausgabedatei)

        # Laufe bis Kollision auftreten sollte
        for _ in range(10):
            sim.schritt()

        # Prüfe, dass Kollision aufgezeichnet wurde
        self.assertGreater(sim.box.gesamt_kollisionen, 0)

    def test_teilchen_bleibt_in_box(self):
        """Teste, dass Teilchen immer innerhalb der Box-Grenzen bleiben."""
        # Führe Simulation mit verschiedenen Anfangsbedingungen aus
        zustaende = np.array([
            [95.0, 95.0, 30.0, 30.0],  # Ecken-gebunden
            [5.0, 5.0, -30.0, -30.0],   # Andere Ecke
            [50.0, 99.0, 0.0, 20.0]     # Obere Wand
        ])

        sim = Simulation(anfangszustaende=zustaende, dt=0.01, ausgabedatei=self.ausgabedatei)

        # Laufe für viele Schritte
        for _ in range(100):
            sim.schritt()

            # Prüfe, dass alle Teilchen in Box bleiben
            for teilchen in sim.teilchen:
                self.assertGreaterEqual(teilchen.x, konst.BOX_MIN_X)
                self.assertLessEqual(teilchen.x, konst.BOX_MAX_X)
                self.assertGreaterEqual(teilchen.y, konst.BOX_MIN_Y)
                self.assertLessEqual(teilchen.y, konst.BOX_MAX_Y)

    def test_datenaufzeichnung(self):
        """Teste, dass Simulationsdaten korrekt aufgezeichnet werden."""
        # Führe ein paar Schritte aus
        n_schritte = 5
        for _ in range(n_schritte):
            self.sim.schritt()

        # Prüfe, dass Daten aufgezeichnet wurden
        self.assertEqual(len(self.sim.datenverwalter.trajektorien_daten['zeit']), n_schritte + 1)  # +1 für Anfang
        self.assertEqual(len(self.sim.datenverwalter.trajektorien_daten['energie']), n_schritte + 1)

        # Prüfe, dass Zeiten korrekt sind
        erwartete_zeiten = [i * self.sim.dt for i in range(n_schritte + 1)]
        tatsaechliche_zeiten = self.sim.datenverwalter.trajektorien_daten['zeit']
        for erw, tats in zip(erwartete_zeiten, tatsaechliche_zeiten):
            self.assertAlmostEqual(erw, tats)

    def test_tatsaechliche_anfangsbedingungen(self):
        """Teste mit den tatsächlichen Projekt-Anfangsbedingungen."""
        sim = Simulation(
            anfangszustaende=konst.ANFANGSZUSTAENDE,
            dt=konst.DT,
            ausgabedatei=self.ausgabedatei
        )

        # Prüfe, dass alle 7 Teilchen erstellt wurden
        self.assertEqual(len(sim.teilchen), 7)

        # Laufe für kurze Zeit
        for _ in range(100):
            sim.schritt()

        # System sollte stabil bleiben
        endenergie = sim.berechne_gesamtenergie()
        self.assertTrue(np.isfinite(endenergie))

        # Alle Teilchen sollten in Box sein
        for teilchen in sim.teilchen:
            self.assertTrue(sim.box.ist_innerhalb(teilchen.position))

    def test_coulomb_abstossung_effekt(self):
        """Teste, dass Coulomb-Abstoßung tatsächlich Trajektorien beeinflusst."""
        # Zwei Teilchen mit Ladung
        zustaende_geladen = np.array([
            [45.0, 50.0, 0.0, 0.0],
            [55.0, 50.0, 0.0, 0.0]
        ])

        sim_geladen = Simulation(
            anfangszustaende=zustaende_geladen,
            dt=0.001,
            ausgabedatei=self.ausgabedatei
        )

        # Führe Simulation aus
        for _ in range(100):
            sim_geladen.schritt()

        # Teilchen sollten sich durch Abstoßung auseinander bewegt haben
        endabstand = sim_geladen.teilchen[0].abstand_zu(sim_geladen.teilchen[1])
        anfangsabstand = 10.0

        self.assertGreater(endabstand, anfangsabstand)

    def test_gravitations_effekt(self):
        """Teste, dass Gravitation Teilchenbewegung beeinflusst."""
        # Einzelnes Teilchen ohne Ladung, nur Gravitation
        zustaende = np.array([[50.0, 80.0, 0.0, 0.0]])  # Hohe Position, keine Geschwindigkeit

        # Setze Ladung temporär auf 0 um Gravitation zu isolieren
        urspruengliche_ladung = konst.LADUNG
        konst.LADUNG = 0.0

        try:
            sim = Simulation(
                anfangszustaende=zustaende,
                dt=0.01,
                ausgabedatei=self.ausgabedatei
            )

            anfangs_y = sim.teilchen[0].y
            anfangs_vy = sim.teilchen[0].vy

            # Laufe ein bisschen
            for _ in range(10):
                sim.schritt()

            # Teilchen sollte gefallen sein (y verringert, vy negativ)
            self.assertLess(sim.teilchen[0].y, anfangs_y)
            self.assertLess(sim.teilchen[0].vy, anfangs_vy)

        finally:
            konst.LADUNG = urspruengliche_ladung

    def test_statistik_berechnung(self):
        """Teste Simulationsstatistiken."""
        # Führe Simulation aus
        for _ in range(50):
            self.sim.schritt()

        statistiken = self.sim.datenverwalter.hole_statistiken()

        # Prüfe, dass Statistiken existieren und Sinn ergeben
        self.assertIn('anfangsenergie', statistiken)
        self.assertIn('endenergie', statistiken)
        self.assertIn('energie_drift', statistiken)
        self.assertIn('relative_drift', statistiken)

        # Energiedrift sollte klein sein
        self.assertLess(abs(statistiken['relative_drift']), 0.01)

    def test_null_masse_teilchen(self):
        """Teste Behandlung von Null-Masse-Teilchen."""
        # Erstelle Teilchen mit Null-Masse
        zustaende = np.array([[50.0, 50.0, 10.0, 10.0]])

        # Setze Masse temporär auf 0
        urspruengliche_masse = konst.MASSE
        konst.MASSE = 0.0

        try:
            sim = Simulation(
                anfangszustaende=zustaende,
                dt=0.001,
                ausgabedatei=self.ausgabedatei
            )

            # Sollte Null-Masse ohne Absturz behandeln
            for _ in range(10):
                erfolg = sim.schritt()
                self.assertTrue(erfolg)

        finally:
            konst.MASSE = urspruengliche_masse

    def test_sehr_kleiner_zeitschritt(self):
        """Teste Simulation mit sehr kleinem Zeitschritt."""
        sim = Simulation(
            anfangszustaende=self.test_zustaende,
            dt=1e-6,
            ausgabedatei=self.ausgabedatei
        )

        # Sollte mit winzigem Zeitschritt funktionieren
        for _ in range(10):
            erfolg = sim.schritt()
            self.assertTrue(erfolg)

        # Energieerhaltung sollte exzellent sein
        endenergie = sim.berechne_gesamtenergie()
        drift = abs(endenergie - sim.anfangsenergie) / abs(sim.anfangsenergie)
        self.assertLess(drift, 1e-8)

    def test_alternative_schritt_methode(self):
        """Teste die alternative Batch-Schritt-Methode."""
        # Erstelle zwei identische Simulationen
        sim1 = Simulation(
            anfangszustaende=self.test_zustaende,
            dt=0.001,
            ausgabedatei=self.ausgabedatei
        )

        sim2 = Simulation(
            anfangszustaende=self.test_zustaende,
            dt=0.001,
            ausgabedatei=self.ausgabedatei + "2"
        )

        # Führe eine mit normalem Schritt, andere mit Alternative aus
        for _ in range(10):
            sim1.schritt()
            sim2.schritt_alternativ_batch()

        # Ergebnisse sollten ähnlich sein (nicht identisch wegen unterschiedlicher Reihenfolge)
        for p1, p2 in zip(sim1.teilchen, sim2.teilchen):
            pos_diff = np.linalg.norm(p1.position - p2.position)
            self.assertLess(pos_diff, 0.1)  # Sollte nah sein


if __name__ == '__main__':
    unittest.main(verbosity=2)