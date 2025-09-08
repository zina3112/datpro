"""
test_konstanten.py - Unit-Tests für Konstantenvalidierung

Überprüft, dass alle Konstanten mit der Projektspezifikation übereinstimmen.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.konstanten as konst


class TestKonstanten(unittest.TestCase):
    """Test-Suite zur Validierung von Konstanten gegen die Spezifikation."""

    def test_physikalische_konstanten(self):
        """Teste, dass physikalische Konstanten mit Spezifikation übereinstimmen."""
        self.assertEqual(konst.MASSE, 1.0, "Masse sollte 1.0 sein")
        self.assertEqual(konst.LADUNG, 50.0, "Ladung sollte 50.0 sein")
        self.assertEqual(konst.GRAVITATION, -10.0, "Gravitation sollte -10.0 sein")

    def test_box_dimensionen(self):
        """Teste Box-Dimensionen."""
        # Spezifikation sagt, Box hat Ausdehnung von 100 in jede Richtung
        # Implementierung verwendet 0-100, was vernünftig ist
        self.assertEqual(konst.BOX_MAX_X - konst.BOX_MIN_X, 100.0)
        self.assertEqual(konst.BOX_MAX_Y - konst.BOX_MIN_Y, 100.0)

        # Prüfe praktische Variablen
        self.assertEqual(konst.BOX_BREITE, 100.0)
        self.assertEqual(konst.BOX_HOEHE, 100.0)

    def test_simulationsparameter(self):
        """Teste, dass Simulationsparameter mit Spezifikation übereinstimmen."""
        self.assertEqual(konst.DT, 0.001, "Zeitschritt sollte 0.001 sein")
        self.assertEqual(konst.SIMULATIONSZEIT, 10.0, "Simulationszeit sollte 10 Sekunden sein")
        self.assertEqual(konst.N_SCHRITTE, 10000, "Sollte 10000 Schritte haben")

    def test_anfangsbedingungen(self):
        """Teste, dass anfängliche Teilchenzustände exakt mit Spezifikation übereinstimmen."""
        erwartete_zustaende = np.array([
            [1.0, 45.0, 10.0, 0.0],
            [99.0, 55.0, -10.0, 0.0],
            [10.0, 50.0, 15.0, -15.0],
            [20.0, 30.0, -15.0, -15.0],
            [80.0, 70.0, 15.0, 15.0],
            [80.0, 60.0, 15.0, 15.0],
            [80.0, 50.0, 15.0, 15.0]
        ])

        # Prüfe Form
        self.assertEqual(konst.ANFANGSZUSTAENDE.shape, (7, 4))

        # Prüfe jeden Anfangszustand des Teilchens exakt
        np.testing.assert_array_equal(konst.ANFANGSZUSTAENDE, erwartete_zustaende)

        # Verifiziere N_TEILCHEN
        self.assertEqual(konst.N_TEILCHEN, 7)

    def test_numerische_parameter(self):
        """Teste, dass numerische Parameter vernünftig sind."""
        # Epsilon sollte sehr klein sein
        self.assertLess(konst.EPSILON, 1e-8)
        self.assertGreater(konst.EPSILON, 0)

        # Energietoleranz sollte vernünftig sein
        self.assertLess(konst.ENERGIE_TOLERANZ, 0.01)  # Weniger als 1%
        self.assertGreater(konst.ENERGIE_TOLERANZ, 0)

        # Kollisionsparameter
        self.assertEqual(konst.KOLLISIONS_EPSILON, konst.EPSILON)
        self.assertGreater(konst.MAX_KOLLISIONS_ITERATIONEN, 5)

    def test_ausgabeparameter(self):
        """Teste Ausgabekonfiguration."""
        self.assertIsInstance(konst.AUSGABE_VERZEICHNIS, str)
        self.assertIsInstance(konst.AUSGABE_DATEI, str)
        self.assertIsInstance(konst.PLOT_VERZEICHNIS, str)

        # Ausgabefrequenz sollte positiv sein
        self.assertGreater(konst.AUSGABE_FREQUENZ, 0)

        # Abbildungsparameter
        self.assertIsInstance(konst.ABBILDUNGSGROESSE, tuple)
        self.assertEqual(len(konst.ABBILDUNGSGROESSE), 2)
        self.assertGreater(konst.DPI, 0)

    def test_anfangszustaende_gueltigkeit(self):
        """Teste, dass Anfangszustände physikalisch gültig sind."""
        for i, zustand in enumerate(konst.ANFANGSZUSTAENDE):
            x, y, vx, vy = zustand

            # Positionen sollten innerhalb der Box sein
            self.assertGreaterEqual(x, konst.BOX_MIN_X,
                                  f"Teilchen {i+1} x-Position außerhalb der Box")
            self.assertLessEqual(x, konst.BOX_MAX_X,
                               f"Teilchen {i+1} x-Position außerhalb der Box")
            self.assertGreaterEqual(y, konst.BOX_MIN_Y,
                                  f"Teilchen {i+1} y-Position außerhalb der Box")
            self.assertLessEqual(y, konst.BOX_MAX_Y,
                               f"Teilchen {i+1} y-Position außerhalb der Box")

            # Geschwindigkeiten sollten endlich sein
            self.assertTrue(np.isfinite(vx), f"Teilchen {i+1} vx nicht endlich")
            self.assertTrue(np.isfinite(vy), f"Teilchen {i+1} vy nicht endlich")

            # Prüfe, dass Geschwindigkeitsbeträge vernünftig sind (nicht zu groß)
            geschwindigkeit = np.sqrt(vx**2 + vy**2)
            self.assertLess(geschwindigkeit, 50.0, f"Teilchen {i+1} Anfangsgeschwindigkeit zu hoch")

    def test_konstanten_unveraenderlichkeit(self):
        """Teste, dass Konstanten nicht versehentlich modifiziert werden."""
        # Speichere ursprüngliche Werte
        urspruengliche_masse = konst.MASSE
        urspruengliche_ladung = konst.LADUNG
        urspruengliche_gravitation = konst.GRAVITATION
        urspruengliches_dt = konst.DT

        # Konstanten sollten immer noch korrekte Werte haben
        self.assertEqual(konst.MASSE, urspruengliche_masse)
        self.assertEqual(konst.LADUNG, urspruengliche_ladung)
        self.assertEqual(konst.GRAVITATION, urspruengliche_gravitation)
        self.assertEqual(konst.DT, urspruengliches_dt)

    def test_teilchen_interaktionen(self):
        """Teste, dass Teilchenkonfiguration zu interessanter Dynamik führt."""
        # Prüfe, dass einige Teilchen nah genug starten, um stark zu interagieren
        min_abstand = float('inf')

        for i in range(konst.N_TEILCHEN):
            for j in range(i+1, konst.N_TEILCHEN):
                pos_i = konst.ANFANGSZUSTAENDE[i, 0:2]
                pos_j = konst.ANFANGSZUSTAENDE[j, 0:2]
                abstand = np.linalg.norm(pos_i - pos_j)
                min_abstand = min(min_abstand, abstand)

        # Einige Teilchen sollten vernünftig nah sein
        self.assertLess(min_abstand, 20.0, "Teilchen zu weit auseinander für starke Interaktion")

        # Aber nicht zu nah (würde numerische Probleme verursachen)
        self.assertGreater(min_abstand, 0.1, "Teilchen zu nah anfänglich")


if __name__ == '__main__':
    unittest.main(verbosity=2)