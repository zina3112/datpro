"""
test_teilchen.py - Unit-Tests für Teilchen-Klasse

Testet Teilcheneigenschaften, Zustandsverwaltung und Energieberechnungen.
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.teilchen import Teilchen
import src.konstanten as konst


class TestTeilchen(unittest.TestCase):
    """Test-Suite für Teilchen-Klasse."""

    def setUp(self):
        """Setze Test-Teilchen auf."""
        self.teilchen1 = Teilchen(x=10.0, y=20.0, vx=5.0, vy=-3.0)
        self.teilchen2 = Teilchen(x=15.0, y=25.0, vx=-2.0, vy=4.0,
                                 masse=2.0, ladung=100.0)

    def test_initialisierung(self):
        """Teste Teilchen-Initialisierung mit Standard- und benutzerdefinierten Werten."""
        # Teste Standard-Masse und -Ladung
        self.assertEqual(self.teilchen1.masse, konst.MASSE)
        self.assertEqual(self.teilchen1.ladung, konst.LADUNG)

        # Teste benutzerdefinierte Masse und Ladung
        self.assertEqual(self.teilchen2.masse, 2.0)
        self.assertEqual(self.teilchen2.ladung, 100.0)

        # Teste Position und Geschwindigkeit
        self.assertEqual(self.teilchen1.x, 10.0)
        self.assertEqual(self.teilchen1.y, 20.0)
        self.assertEqual(self.teilchen1.vx, 5.0)
        self.assertEqual(self.teilchen1.vy, -3.0)

    def test_zustandsvektor(self):
        """Teste Zustandsvektor-Darstellung."""
        erwarteter_zustand = np.array([10.0, 20.0, 5.0, -3.0])
        np.testing.assert_array_equal(self.teilchen1.zustand, erwarteter_zustand)

        # Teste, dass Zustand ein NumPy-Array ist
        self.assertIsInstance(self.teilchen1.zustand, np.ndarray)
        self.assertEqual(self.teilchen1.zustand.dtype, np.float64)

    def test_positions_eigenschaft(self):
        """Teste Positions-Eigenschaft Getter und Setter."""
        # Teste Getter
        pos = self.teilchen1.position
        np.testing.assert_array_equal(pos, np.array([10.0, 20.0]))

        # Teste Setter
        neue_pos = np.array([30.0, 40.0])
        self.teilchen1.position = neue_pos
        np.testing.assert_array_equal(self.teilchen1.position, neue_pos)
        self.assertEqual(self.teilchen1.x, 30.0)
        self.assertEqual(self.teilchen1.y, 40.0)

    def test_geschwindigkeits_eigenschaft(self):
        """Teste Geschwindigkeits-Eigenschaft Getter und Setter."""
        # Teste Getter
        ges = self.teilchen1.geschwindigkeit
        np.testing.assert_array_equal(ges, np.array([5.0, -3.0]))

        # Teste Setter
        neue_ges = np.array([7.0, -8.0])
        self.teilchen1.geschwindigkeit = neue_ges
        np.testing.assert_array_equal(self.teilchen1.geschwindigkeit, neue_ges)
        self.assertEqual(self.teilchen1.vx, 7.0)
        self.assertEqual(self.teilchen1.vy, -8.0)

    def test_kinetische_energie(self):
        """Teste Berechnung der kinetischen Energie."""
        # KE = 0.5 * m * (vx^2 + vy^2)
        # Für teilchen1: 0.5 * 1.0 * (5^2 + 3^2) = 0.5 * 34 = 17.0
        erwartete_ke = 0.5 * self.teilchen1.masse * (5.0**2 + 3.0**2)
        self.assertAlmostEqual(self.teilchen1.kinetische_energie(), erwartete_ke)

        # Für teilchen2 mit masse=2.0
        erwartete_ke2 = 0.5 * 2.0 * ((-2.0)**2 + 4.0**2)
        self.assertAlmostEqual(self.teilchen2.kinetische_energie(), erwartete_ke2)

    def test_gravitationelle_potentielle_energie(self):
        """Teste Berechnung der gravitationellen potentiellen Energie."""
        # PE = -m * g * y wobei g negativ ist
        # Für teilchen1: -1.0 * (-10.0) * 20.0 = 200.0
        erwartete_pe = -self.teilchen1.masse * konst.GRAVITATION * self.teilchen1.y
        self.assertAlmostEqual(self.teilchen1.potentielle_energie_gravitation(), erwartete_pe)

        # Verifiziere, dass sie mit Höhe zunimmt
        self.teilchen1.position = np.array([10.0, 30.0])
        neue_pe = self.teilchen1.potentielle_energie_gravitation()
        self.assertGreater(neue_pe, erwartete_pe)

    def test_abstand_zu(self):
        """Teste Abstandsberechnung zwischen Teilchen."""
        abstand = self.teilchen1.abstand_zu(self.teilchen2)
        # Abstand = sqrt((15-10)^2 + (25-20)^2) = sqrt(25 + 25) = sqrt(50)
        erwarteter_abstand = np.sqrt(50)
        self.assertAlmostEqual(abstand, erwarteter_abstand)

        # Teste, dass Abstand symmetrisch ist
        abstand_umgekehrt = self.teilchen2.abstand_zu(self.teilchen1)
        self.assertAlmostEqual(abstand, abstand_umgekehrt)

        # Teste Null-Abstand für gleiches Teilchen
        self.assertAlmostEqual(self.teilchen1.abstand_zu(self.teilchen1), 0.0)

    def test_verschiebung_zu(self):
        """Teste Verschiebungsvektor-Berechnung."""
        verschiebung = self.teilchen1.verschiebung_zu(self.teilchen2)
        # Verschiebung von teilchen2 zu teilchen1: (10-15, 20-25) = (-5, -5)
        erwartete = np.array([-5.0, -5.0])
        np.testing.assert_array_almost_equal(verschiebung, erwartete)

        # Teste umgekehrte Verschiebung
        verschiebung_umgekehrt = self.teilchen2.verschiebung_zu(self.teilchen1)
        np.testing.assert_array_almost_equal(verschiebung_umgekehrt, -erwartete)

    def test_aktualisiere_zustand(self):
        """Teste Zustandsaktualisierungs-Funktionalität."""
        neuer_zustand = np.array([100.0, 200.0, 10.0, -20.0])
        self.teilchen1.aktualisiere_zustand(neuer_zustand)

        np.testing.assert_array_equal(self.teilchen1.zustand, neuer_zustand)
        self.assertEqual(self.teilchen1.x, 100.0)
        self.assertEqual(self.teilchen1.y, 200.0)
        self.assertEqual(self.teilchen1.vx, 10.0)
        self.assertEqual(self.teilchen1.vy, -20.0)

        # Teste ungültigen Zustandsvektor
        with self.assertRaises(ValueError):
            self.teilchen1.aktualisiere_zustand(np.array([1.0, 2.0]))  # Falsche Größe

    def test_kopiere(self):
        """Teste tiefe Kopie des Teilchens."""
        kopie = self.teilchen1.kopiere()

        # Teste, dass Werte gleich sind
        np.testing.assert_array_equal(kopie.zustand, self.teilchen1.zustand)
        self.assertEqual(kopie.masse, self.teilchen1.masse)
        self.assertEqual(kopie.ladung, self.teilchen1.ladung)

        # Teste, dass es eine tiefe Kopie ist (Modifikation der Kopie beeinflusst nicht Original)
        kopie.aktualisiere_zustand(np.array([0.0, 0.0, 0.0, 0.0]))
        self.assertNotEqual(kopie.x, self.teilchen1.x)

    def test_eindeutige_teilchen_ids(self):
        """Teste, dass jedes Teilchen eine eindeutige ID erhält."""
        p1 = Teilchen(0, 0, 0, 0)
        p2 = Teilchen(0, 0, 0, 0)
        p3 = Teilchen(0, 0, 0, 0)

        # Alle IDs sollten unterschiedlich sein
        self.assertNotEqual(p1.teilchen_id, p2.teilchen_id)
        self.assertNotEqual(p2.teilchen_id, p3.teilchen_id)
        self.assertNotEqual(p1.teilchen_id, p3.teilchen_id)

    def test_kollisionsverfolgung(self):
        """Teste Kollisionszähler-Verfolgung."""
        self.assertEqual(self.teilchen1.kollisionszaehler, 0)
        self.assertEqual(self.teilchen1.letzte_kollisionszeit, -1.0)

        # Simuliere Kollision
        self.teilchen1.kollisionszaehler += 1
        self.teilchen1.letzte_kollisionszeit = 5.5

        self.assertEqual(self.teilchen1.kollisionszaehler, 1)
        self.assertEqual(self.teilchen1.letzte_kollisionszeit, 5.5)

    def test_string_darstellung(self):
        """Teste String-Darstellungen."""
        str_darst = str(self.teilchen1)
        self.assertIn("Teilchen", str_darst)
        self.assertIn("10.00", str_darst)  # x-Position
        self.assertIn("20.00", str_darst)  # y-Position

        repr_str = repr(self.teilchen1)
        self.assertIn("Teilchen(", repr_str)
        self.assertIn("x=10.0", repr_str)


if __name__ == '__main__':
    unittest.main(verbosity=2)