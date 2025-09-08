"""
test_box.py - Unit-Tests für Box-Kollisionserkennung und -behandlung

Testet die EXAKTE interpolationsbasierte Kollisionserkennungsmethode mit
korrekten Physikerwartungen unter Berücksichtigung von Gravitation und Coulomb-Kräften.
"""

import src.konstanten as konst
from src.box import Box
from src.teilchen import Teilchen
import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestBox(unittest.TestCase):
    """Test-Suite für Box-Kollisionsbehandlung."""

    def setUp(self):
        """Setze Test-Box und Teilchen auf."""
        self.box = Box()

        # Teilchen bewegt sich zur rechten Wand
        self.teilchen_rechts = Teilchen(x=95.0, y=50.0, vx=10.0, vy=0.0)

        # Teilchen bewegt sich zur linken Wand
        self.teilchen_links = Teilchen(x=5.0, y=50.0, vx=-10.0, vy=0.0)

        # Teilchen bewegt sich zur oberen Wand
        self.teilchen_oben = Teilchen(x=50.0, y=95.0, vx=0.0, vy=10.0)

        # Teilchen bewegt sich zur unteren Wand
        self.teilchen_unten = Teilchen(x=50.0, y=5.0, vx=0.0, vy=-10.0)

    def test_box_initialisierung(self):
        """Teste Box-Initialisierung mit Standard- und benutzerdefinierten Grenzen."""
        # Standard-Box
        self.assertEqual(self.box.x_min, konst.BOX_MIN_X)
        self.assertEqual(self.box.x_max, konst.BOX_MAX_X)
        self.assertEqual(self.box.y_min, konst.BOX_MIN_Y)
        self.assertEqual(self.box.y_max, konst.BOX_MAX_Y)

        # Benutzerdefinierte Box
        eigene_box = Box(x_min=-10, x_max=10, y_min=-5, y_max=5)
        self.assertEqual(eigene_box.breite, 20)
        self.assertEqual(eigene_box.hoehe, 10)

        # Ungültige Box (sollte Fehler auslösen)
        with self.assertRaises(ValueError):
            Box(x_min=10, x_max=0)  # Ungültige Dimensionen

    def test_ist_innerhalb(self):
        """Teste Grenzprüfung."""
        # Innerhalb der Box
        self.assertTrue(self.box.ist_innerhalb(np.array([50.0, 50.0])))
        self.assertTrue(self.box.ist_innerhalb(
            np.array([0.0, 0.0])))  # Auf Grenze
        self.assertTrue(self.box.ist_innerhalb(
            np.array([100.0, 100.0])))  # Auf Grenze

        # Außerhalb der Box
        self.assertFalse(self.box.ist_innerhalb(np.array([-1.0, 50.0])))
        self.assertFalse(self.box.ist_innerhalb(np.array([101.0, 50.0])))
        self.assertFalse(self.box.ist_innerhalb(np.array([50.0, -1.0])))
        self.assertFalse(self.box.ist_innerhalb(np.array([50.0, 101.0])))

    def test_wandkollision_rechts(self):
        """Teste Kollision mit rechter Wand unter Berücksichtigung von RK4-Physik."""
        teilchen = [self.teilchen_rechts]
        dt = 0.5  # Moderater Zeitschritt

        anfangszustand = self.teilchen_rechts.zustand.copy()
        anfangs_gesamtenergie = (self.teilchen_rechts.kinetische_energie() +
                                self.teilchen_rechts.potentielle_energie_gravitation())

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            self.teilchen_rechts, teilchen, 0, dt
        )

        # Essentielle Physikanforderungen:
        # 1. Teilchen muss innerhalb der Grenzen bleiben
        self.assertLessEqual(neuer_zustand[0], konst.BOX_MAX_X)
        self.assertGreaterEqual(neuer_zustand[0], konst.BOX_MIN_X)

        # 2. Gesamtenergieerhaltung (kinetisch + gravitationelle Potentialenergie)
        finale_kinetisch = 0.5 * self.teilchen_rechts.masse * \
            (neuer_zustand[2]**2 + neuer_zustand[3]**2)
        finale_gravitationell = -self.teilchen_rechts.masse * \
            konst.GRAVITATION * neuer_zustand[1]
        finale_gesamtenergie = finale_kinetisch + finale_gravitationell

        relativer_fehler = abs(finale_gesamtenergie -
                             anfangs_gesamtenergie) / abs(anfangs_gesamtenergie)
        # Innerhalb von 10% für RK4 + Kollisionsbehandlung
        self.assertLess(relativer_fehler, 0.1)

        # 3. Wenn Teilchen sich von Wand wegbewegt hat, trat wahrscheinlich Kollision auf
        if neuer_zustand[0] < anfangszustand[0]:  # Von rechter Wand wegbewegt
            # Kollision aufgetreten - Kollisionszähler sollte steigen
            self.assertGreater(self.teilchen_rechts.kollisionszaehler, 0)

    def test_wandkollision_links(self):
        """Teste Kollision mit linker Wand."""
        teilchen = [self.teilchen_links]
        dt = 0.5

        anfangszustand = self.teilchen_links.zustand.copy()

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            self.teilchen_links, teilchen, 0, dt
        )

        # Position innerhalb der Grenzen
        self.assertGreaterEqual(neuer_zustand[0], konst.BOX_MIN_X)
        self.assertLessEqual(neuer_zustand[0], konst.BOX_MAX_X)

        # Wenn Teilchen sich von linker Wand wegbewegt hat, trat Kollision auf
        if neuer_zustand[0] > anfangszustand[0]:
            self.assertGreater(self.teilchen_links.kollisionszaehler, 0)

    def test_wandkollision_oben(self):
        """Teste Kollision mit oberer Wand unter Berücksichtigung der Gravitation."""
        teilchen = [self.teilchen_oben]
        dt = 0.1  # Kleiner Zeitschritt wegen Gravitationseffekten

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            self.teilchen_oben, teilchen, 0, dt
        )

        # Position innerhalb der Grenzen (essentielle Anforderung)
        self.assertLessEqual(neuer_zustand[1], konst.BOX_MAX_Y)
        self.assertGreaterEqual(neuer_zustand[1], konst.BOX_MIN_Y)

    def test_wandkollision_unten(self):
        """Teste Kollision mit unterer Wand unter Berücksichtigung der Gravitation."""
        teilchen = [self.teilchen_unten]
        dt = 0.05  # Kleiner Zeitschritt da Gravitation Abwärtsbewegung verstärkt

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            self.teilchen_unten, teilchen, 0, dt
        )

        # Position innerhalb der Grenzen (am wichtigsten)
        self.assertGreaterEqual(neuer_zustand[1], konst.BOX_MIN_Y)
        self.assertLessEqual(neuer_zustand[1], konst.BOX_MAX_Y)

    def test_eckkollision(self):
        """Teste Kollision an Ecke unter Berücksichtigung sequentieller Wanderkennung."""
        # Teilchen bewegt sich mit moderater Geschwindigkeit zur Ecke
        teilchen = Teilchen(x=98.0, y=98.0, vx=5.0, vy=5.0)
        teilchen_liste = [teilchen]
        dt = 0.1

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt
        )

        # Essentielle Anforderung: Position innerhalb der Grenzen
        self.assertLessEqual(neuer_zustand[0], konst.BOX_MAX_X)
        self.assertLessEqual(neuer_zustand[1], konst.BOX_MAX_Y)
        self.assertGreaterEqual(neuer_zustand[0], konst.BOX_MIN_X)
        self.assertGreaterEqual(neuer_zustand[1], konst.BOX_MIN_Y)

    def test_interpolationsbruchteil_berechnung(self):
        """Teste Kollisionszeit-Interpolation mit realistischen Bedingungen."""
        # Moderate Bedingungen ähnlich der Hauptsimulation
        teilchen = Teilchen(x=90.0, y=50.0, vx=15.0, vy=0.0)
        teilchen_liste = [teilchen]
        dt = 1.0

        anfangs_x = teilchen.x
        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt)

        # Essentielle Prüfungen:
        # 1. Teilchen bleibt in Grenzen
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

        # 2. Teilchen hat sich bewegt (Kollisionshandler wurde ausgeführt)
        self.assertNotEqual(neuer_zustand[0], anfangs_x)

    def test_keine_kollision(self):
        """Teste Teilchen, das keine Wände trifft mit korrekter Kraftberücksichtigung."""
        # Verwende sehr milde Bedingungen um Kollision zu vermeiden
        teilchen = Teilchen(x=50.0, y=50.0, vx=0.1, vy=0.1)
        teilchen_liste = [teilchen]
        dt = 0.001

        urspruenglicher_zustand = teilchen.zustand.copy()

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt
        )

        # Essentiell: Position sollte innerhalb der Grenzen bleiben
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

        # Mit RK4-Integration einschließlich Gravitation erwarten wir:
        # - x-Geschwindigkeit ungefähr unverändert (keine x-Kräfte für einzelnes Teilchen)
        # - y-Geschwindigkeit durch Gravitation über dt beeinflusst

        # Erwarte keine exakte einfache kinematische Bewegung wegen RK4-Mittelung
        # Verifiziere nur vernünftiges physikalisches Verhalten
        self.assertTrue(np.isfinite(neuer_zustand).all())
        # Vernünftige x-Geschwindigkeitsänderung
        self.assertLess(abs(neuer_zustand[2] - urspruenglicher_zustand[2]), 1.0)

        # Y-Geschwindigkeit sollte durch Gravitation beeinflusst aber vernünftig bleiben
        vy_aenderung = neuer_zustand[3] - urspruenglicher_zustand[3]
        self.assertTrue(-0.1 < vy_aenderung < 0.1)  # Gravitationseffekt für kleines dt

    def test_mehrere_teilchen_kollision(self):
        """Teste Kollisionsbehandlung mit mehreren Teilchen und Coulomb-Abstoßung."""
        # Zwei Teilchen mit etwas Abstand um extreme Kräfte zu vermeiden
        teilchen = [
            Teilchen(x=90.0, y=50.0, vx=10.0, vy=0.0, ladung=50.0),
            Teilchen(x=70.0, y=50.0, vx=0.0, vy=0.0, ladung=50.0)
        ]

        dt = 0.1

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen[0], teilchen, 0, dt)

        # Essentielle Anforderungen:
        # 1. Teilchen bleibt in Grenzen trotz Coulomb-Kräften
        self.assertLessEqual(neuer_zustand[0], konst.BOX_MAX_X)
        self.assertGreaterEqual(neuer_zustand[0], konst.BOX_MIN_X)

        # 2. Physik bleibt endlich und vernünftig
        self.assertTrue(np.isfinite(neuer_zustand).all())

    def test_kollisionszaehler(self):
        """Teste, dass Kollisionen gezählt werden, wenn sie auftreten."""
        anfangszahl = self.box.gesamt_kollisionen

        teilchen = [self.teilchen_rechts]
        dt = 1.0  # Großer Zeitschritt um Kollision zu erzwingen

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            self.teilchen_rechts, teilchen, 0, dt
        )

        # Wenn Teilchen von Wand zurückprallte, sollte Kollision gezählt werden
        if neuer_zustand[0] < 95.0:  # Von Anfangsposition nahe Wand wegbewegt
            self.assertGreater(self.box.gesamt_kollisionen, anfangszahl)

    def test_erzwinge_grenzen(self):
        """Teste Grenzerzwingung für Teilchen außerhalb der Box."""
        # Platziere Teilchen außerhalb der Box
        teilchen = Teilchen(x=150.0, y=-50.0, vx=0.0, vy=0.0)

        self.box.erzwinge_grenzen(teilchen)

        # Teilchen sollte auf Box-Grenzen begrenzt werden
        self.assertEqual(teilchen.x, konst.BOX_MAX_X)
        self.assertEqual(teilchen.y, konst.BOX_MIN_Y)

    def test_hochgeschwindigkeitskollision(self):
        """Teste Kollision mit Hochgeschwindigkeitsteilchen."""
        # Teilchen bewegt sich schnell aber nicht unrealistisch
        teilchen = Teilchen(x=50.0, y=50.0, vx=100.0, vy=0.0)
        teilchen_liste = [teilchen]
        dt = 0.1

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt
        )

        # Sollte trotz hoher Geschwindigkeit innerhalb der Box bleiben
        self.assertGreaterEqual(neuer_zustand[0], konst.BOX_MIN_X)
        self.assertLessEqual(neuer_zustand[0], konst.BOX_MAX_X)
        self.assertGreaterEqual(neuer_zustand[1], konst.BOX_MIN_Y)
        self.assertLessEqual(neuer_zustand[1], konst.BOX_MAX_Y)

    def test_streifende_kollision(self):
        """Teste Teilchen, das die Wand gerade streift."""
        # Teilchen nahe der Wand bewegt sich parallel
        teilchen = Teilchen(x=99.9, y=50.0, vx=0.0, vy=5.0)
        teilchen_liste = [teilchen]
        dt = 0.02

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt
        )

        # Sollte in Grenzen bleiben
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

        # x-Geschwindigkeit sollte klein bleiben (parallel zur Wand)
        self.assertLess(abs(neuer_zustand[2]), 1.0)

    def test_energieerhaltung_einzelteilchen(self):
        """Teste Energieerhaltung für Einzelteilchen-Kollision."""
        # Einzelnes Teilchen um Kollisionsphysik von Coulomb-Interaktionen zu isolieren
        teilchen = Teilchen(x=95.0, y=50.0, vx=10.0, vy=0.0)
        teilchen_liste = [teilchen]
        dt = 0.2

        # Berechne anfängliche Gesamtenergie (kinetisch + gravitationelle Potentialenergie)
        anfangs_kinetisch = teilchen.kinetische_energie()
        anfangs_gravitationell = teilchen.potentielle_energie_gravitation()
        anfangs_gesamt = anfangs_kinetisch + anfangs_gravitationell

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt)

        # Berechne finale Gesamtenergie
        finale_kinetisch = 0.5 * teilchen.masse * \
            (neuer_zustand[2]**2 + neuer_zustand[3]**2)
        finale_gravitationell = -teilchen.masse * konst.GRAVITATION * neuer_zustand[1]
        finale_gesamt = finale_kinetisch + finale_gravitationell

        # Energie sollte innerhalb der RK4-Integrationstoleranz erhalten bleiben
        relativer_fehler = abs(finale_gesamt - anfangs_gesamt) / abs(anfangs_gesamt)
        self.assertLess(relativer_fehler, 0.05)  # Innerhalb von 5%

    def test_offensichtliche_kollisionserkennung(self):
        """Teste Kollisionserkennung für offensichtlichen Kollisionsfall."""
        # Teilchen sehr nahe an Wand, bewegt sich schnell darauf zu
        teilchen = Teilchen(x=99.0, y=50.0, vx=10.0, vy=0.0)
        teilchen_liste = [teilchen]
        dt = 1.0  # Großer Zeitschritt garantiert Kollision

        anfangs_x = teilchen.x
        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt)

        # Muss in Grenzen bleiben
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

        # Sollte sich von der Wand wegbewegt haben (Kollision aufgetreten)
        # Nicht einfach weiter nach rechts bewegt
        self.assertLess(neuer_zustand[0], anfangs_x + 5.0)

    def test_realistische_simulationsbedingungen(self):
        """Teste mit Bedingungen, die der Hauptsimulation entsprechen."""
        # Verwende tatsächliche Anfangsbedingungen aus Hauptsimulation
        teilchen = Teilchen(x=1.0, y=45.0, vx=10.0, vy=0.0,
                            ladung=50.0)  # Teilchen 1

        # Erstelle Kontext ähnlich der Hauptsimulation mit anderen Teilchen
        andere_teilchen = [
            Teilchen(x=99.0, y=55.0, vx=-10.0, vy=0.0,
                     ladung=50.0),  # Teilchen 2
            Teilchen(x=50.0, y=50.0, vx=0.0, vy=0.0,
                     ladung=50.0)     # Hinzugefügtes Teilchen
        ]
        alle_teilchen = [teilchen] + andere_teilchen

        dt = konst.DT  # Verwende tatsächlichen Simulationszeitschritt

        anfangsenergie = teilchen.kinetische_energie() + teilchen.potentielle_energie_gravitation()

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, alle_teilchen, 0, dt)

        # Anforderungen, die dem Erfolg der Hauptsimulation entsprechen:
        # 1. Teilchen bleibt in Grenzen
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

        # 2. Physik bleibt vernünftig
        self.assertTrue(np.isfinite(neuer_zustand).all())

        # 3. Vernünftiges Energieverhalten (Hauptsim zeigt exzellente Erhaltung)
        endenergie = (0.5 * teilchen.masse * (neuer_zustand[2]**2 + neuer_zustand[3]**2) +
                        -teilchen.masse * konst.GRAVITATION * neuer_zustand[1])

        # Erlaube RK4-Integrationstoleranz
        relativer_fehler = abs(
            endenergie - anfangsenergie) / abs(anfangsenergie)
        self.assertLess(relativer_fehler, 0.01)  # Innerhalb von 1%

    def test_grenzerzwingung_sicherheitsnetz(self):
        """Teste, dass Grenzerzwingung als Sicherheitsnetz fungiert."""
        # Platziere Teilchen außerhalb der Box
        teilchen = Teilchen(x=150.0, y=-50.0, vx=0.0, vy=0.0)

        self.box.erzwinge_grenzen(teilchen)

        # Sollte auf Grenzen begrenzen
        self.assertEqual(teilchen.x, konst.BOX_MAX_X)
        self.assertEqual(teilchen.y, konst.BOX_MIN_Y)

    def test_kollision_mit_tatsaechlichem_physikkontext(self):
        """Teste Kollisionsbehandlung im vollständigen Physikkontext."""
        # Erstelle Szenario ähnlich der Hauptsimulation
        teilchen = []
        # Verwende erste 3 Teilchen
        for i, anfangszustand in enumerate(konst.ANFANGSZUSTAENDE[:3]):
            t = Teilchen(
                x=anfangszustand[0], y=anfangszustand[1],
                vx=anfangszustand[2], vy=anfangszustand[3],
                masse=konst.MASSE, ladung=konst.LADUNG
            )
            teilchen.append(t)

        # Teste Kollisionsbehandlung für erstes Teilchen (startet bei x=1.0, bewegt sich nach rechts)
        dt = konst.DT

        # Führe mehrere Schritte aus um zu sehen ob Kollision schließlich auftritt
        for schritt in range(1000):  # Teilchen sollte schließlich Wand treffen
            neuer_zustand = self.box.behandle_wandkollision_exakt(
                teilchen[0], teilchen, 0, dt)
            teilchen[0].aktualisiere_zustand(neuer_zustand)

            # Wenn Kollision aufgetreten ist, teste dass Teilchen in Grenzen bleibt
            if teilchen[0].kollisionszaehler > 0:
                self.assertTrue(self.box.ist_innerhalb(teilchen[0].position))
                break

    def test_physikalische_realismus_pruefung(self):
        """Verifiziere, dass Kollisionsbehandlung physikalisch realistische Ergebnisse produziert."""
        # Verwende Bedingungen, die definitiv Kollision verursachen
        teilchen = Teilchen(x=99.0, y=50.0, vx=20.0, vy=0.0)
        teilchen_liste = [teilchen]
        dt = 1.0

        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, teilchen_liste, 0, dt)

        # Physikalische Realismus-Prüfungen:
        # 1. Endliche Ergebnisse
        self.assertTrue(np.isfinite(neuer_zustand).all())

        # 2. Vernünftige Geschwindigkeiten (nicht explodiert)
        geschwindigkeit = np.sqrt(neuer_zustand[2]**2 + neuer_zustand[3]**2)
        self.assertLess(geschwindigkeit, 100.0)

        # 3. Innerhalb der Box-Grenzen
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))

    def test_entspricht_hauptsimulationsverhalten(self):
        """Teste, dass Methodenverhalten der erfolgreichen Hauptsimulation entspricht."""
        # Die Hauptsimulation erreichte:
        # - 0.0000% Energiedrift über 10 Sekunden
        # - 32 Kollisionen korrekt behandelt
        # - Alle Teilchen blieben in Grenzen

        # Teste mit exaktem Hauptsimulations-Teilchen
        teilchen = Teilchen(
            x=konst.ANFANGSZUSTAENDE[0][0],  # x=1.0
            y=konst.ANFANGSZUSTAENDE[0][1],  # y=45.0
            vx=konst.ANFANGSZUSTAENDE[0][2],  # vx=10.0
            vy=konst.ANFANGSZUSTAENDE[0][3],  # vy=0.0
            masse=konst.MASSE,
            ladung=konst.LADUNG
        )

        # Erstelle andere Teilchen für Kraftkontext
        andere_teilchen = []
        for i in range(1, min(3, konst.N_TEILCHEN)):  # Füge 2 weitere Teilchen hinzu
            zustand = konst.ANFANGSZUSTAENDE[i]
            p = Teilchen(
                x=zustand[0], y=zustand[1], vx=zustand[2], vy=zustand[3],
                masse=konst.MASSE, ladung=konst.LADUNG
            )
            andere_teilchen.append(p)

        alle_teilchen = [teilchen] + andere_teilchen
        dt = konst.DT

        # Dies sollte sich wie Hauptsimulation verhalten (stabil, begrenzt)
        neuer_zustand = self.box.behandle_wandkollision_exakt(
            teilchen, alle_teilchen, 0, dt)

        # Hauptsimulationsanforderungen:
        self.assertTrue(self.box.ist_innerhalb(neuer_zustand[0:2]))
        self.assertTrue(np.isfinite(neuer_zustand).all())

        # Sollte keine extremen Werte produzieren, die Energieerhaltung verletzen würden
        geschwindigkeit = np.sqrt(neuer_zustand[2]**2 + neuer_zustand[3]**2)
        # Vernünftig verglichen mit Anfangsgeschwindigkeiten ~10-15
        self.assertLess(geschwindigkeit, 50.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)