"""
test_datenverwalter.py - Unit-Tests für Datenaufzeichnung und I/O

Testet CSV-Ausgabe, Trajektorienspeicherung und Datenabruf mit korrekter
Teilchenanzahl und Datenstruktur-Erwartungen.
"""

import src.konstanten as konst
from src.teilchen import Teilchen
from src.datenverwalter import Datenverwalter
import unittest
import numpy as np
import sys
import os
import tempfile
import csv
import shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestDatenverwalter(unittest.TestCase):
    """Test-Suite für Datenverwalter-Klasse."""

    def setUp(self):
        """Setze Test-Datenverwalter mit korrekter Teilchenanzahl auf."""
        # Erstelle temporäres Verzeichnis für Testdateien
        self.test_verz = tempfile.mkdtemp()
        self.test_datei = os.path.join(self.test_verz, "test_daten.csv")

        self.datenverwalter = Datenverwalter(self.test_datei)

        # Erstelle Test-Teilchen passend zu Systemerwartungen (konst.N_TEILCHEN)
        self.teilchen = []
        for i in range(konst.N_TEILCHEN):
            x = 10.0 + i * 10.0
            y = 20.0 + i * 5.0
            vx = 5.0 - i * 1.0
            vy = -3.0 + i * 0.5
            teilchen = Teilchen(x=x, y=y, vx=vx, vy=vy)
            self.teilchen.append(teilchen)

    def tearDown(self):
        """Räume temporäre Dateien auf."""
        if os.path.exists(self.test_verz):
            shutil.rmtree(self.test_verz)

    def test_initialisierung(self):
        """Teste Datenverwalter-Initialisierung."""
        self.assertEqual(self.datenverwalter.ausgabedatei, self.test_datei)
        self.assertEqual(self.datenverwalter.geschriebene_datensaetze, 0)
        self.assertEqual(len(self.datenverwalter.trajektorien_daten['zeit']), 0)

    def test_header_generierung(self):
        """Teste CSV-Header-Generierung entspricht Spezifikation."""
        erwarteter_header = ['t', 'E_total']
        for i in range(konst.N_TEILCHEN):
            erwarteter_header.extend(
                [f'x{i + 1}', f'y{i + 1}', f'vx{i + 1}', f'vy{i + 1}'])

        # Verifiziere Header-Struktur
        self.assertEqual(self.datenverwalter.header, erwarteter_header)
        self.assertEqual(len(self.datenverwalter.header),
                         2 + 4 * konst.N_TEILCHEN)

        # Verifiziere, dass spezifische Spalten existieren
        self.assertEqual(self.datenverwalter.header[:2], ['t', 'E_total'])
        self.assertIn('x1', self.datenverwalter.header)
        self.assertIn('vy7', self.datenverwalter.header)

    def test_erfasse_zustand(self):
        """Teste Erfassung des Simulationszustands."""
        zeit = 1.5
        energie = 100.5

        self.datenverwalter.erfasse_zustand(zeit, energie, self.teilchen)

        # Verifiziere Aufzeichnung
        self.assertEqual(len(self.datenverwalter.datenpuffer), 1)
        self.assertEqual(self.datenverwalter.geschriebene_datensaetze, 1)

        # Prüfe Zeit und Energie
        self.assertEqual(self.datenverwalter.trajektorien_daten['zeit'][0], zeit)
        self.assertEqual(
            self.datenverwalter.trajektorien_daten['energie'][0], energie)

        # Prüfe Teilchendatenstruktur
        teilchen_daten = self.datenverwalter.trajektorien_daten['teilchen'][0]
        self.assertEqual(len(teilchen_daten), konst.N_TEILCHEN)

        # Verifiziere spezifische Teilchenwerte
        self.assertEqual(teilchen_daten[0]['x'], 10.0)
        self.assertEqual(teilchen_daten[0]['y'], 20.0)
        self.assertEqual(teilchen_daten[1]['vx'], 4.0)
        self.assertEqual(teilchen_daten[6]['x'], 70.0)

    def test_speichern_in_csv(self):
        """Teste Speichern vollständiger Daten in CSV-Datei."""
        # Erfasse Daten für mehrere Zeitschritte
        for t in range(3):
            self.datenverwalter.erfasse_zustand(
                zeit=t * 0.1,
                gesamtenergie=100.0 - t,
                teilchen=self.teilchen
            )

        # Speichere in Datei
        self.datenverwalter.speichern()

        # Verifiziere, dass Datei existiert
        self.assertTrue(os.path.exists(self.test_datei))

        # Lese und verifiziere CSV-Struktur
        with open(self.test_datei, 'r') as f:
            leser = csv.reader(f)
            zeilen = list(leser)

        # Prüfe Struktur entspricht Spezifikation
        self.assertEqual(len(zeilen), 4)  # Header + 3 Datenzeilen
        self.assertEqual(len(zeilen[0]), 2 + 4 * konst.N_TEILCHEN)

        # Verifiziere Header und Daten
        self.assertEqual(zeilen[0][0], 't')
        self.assertEqual(zeilen[0][1], 'E_total')
        self.assertEqual(float(zeilen[1][0]), 0.0)
        self.assertEqual(float(zeilen[2][0]), 0.1)
        self.assertEqual(float(zeilen[3][1]), 98.0)

    def test_speichern_inkrementell(self):
        """Teste inkrementelles Datenspeichern."""
        # Erfasse und speichere inkrementell
        self.datenverwalter.erfasse_zustand(0.0, 100.0, self.teilchen)
        self.datenverwalter.speichern_inkrementell()

        # Puffer sollte nach Speichern gelöscht sein
        self.assertEqual(len(self.datenverwalter.datenpuffer), 0)

        # Erfasse mehr Daten
        self.datenverwalter.erfasse_zustand(0.1, 99.0, self.teilchen)
        self.datenverwalter.speichern_inkrementell()

        # Datei sollte beide Datensätze enthalten
        with open(self.test_datei, 'r') as f:
            leser = csv.reader(f)
            zeilen = list(leser)

        self.assertEqual(len(zeilen), 3)  # Header + 2 Datenzeilen

    def test_hole_teilchen_trajektorie(self):
        """Teste Abruf individueller Teilchentrajektorien."""
        # Erfasse Zeitschritte mit sich ändernden Positionen
        for t in range(5):
            for i, teilchen in enumerate(self.teilchen):
                teilchen.position = np.array([
                    10.0 + i * 10.0 + t,
                    20.0 + i * 5.0 + t * 0.5
                ])

            self.datenverwalter.erfasse_zustand(t * 0.1, 100.0, self.teilchen)

        # Teste erste Teilchentrajektorie
        x_pos, y_pos = self.datenverwalter.hole_teilchen_trajektorie(0)

        self.assertEqual(len(x_pos), 5)
        self.assertEqual(len(y_pos), 5)

        # Verifiziere Trajektorienwerte
        self.assertEqual(x_pos[0], 10.0)
        self.assertEqual(x_pos[4], 14.0)
        self.assertEqual(y_pos[0], 20.0)
        self.assertEqual(y_pos[4], 22.0)

        # Teste letzte Teilchentrajektorie
        x_pos_letzt, y_pos_letzt = self.datenverwalter.hole_teilchen_trajektorie(
            konst.N_TEILCHEN - 1)
        self.assertEqual(x_pos_letzt[0], 70.0)

        # Teste ungültigen Index
        with self.assertRaises(ValueError):
            self.datenverwalter.hole_teilchen_trajektorie(100)

    def test_hole_alle_trajektorien(self):
        """Teste Abruf aller Teilchentrajektorien."""
        # Erfasse Daten mit allen Teilchen
        self.datenverwalter.erfasse_zustand(0.0, 100.0, self.teilchen)
        self.datenverwalter.erfasse_zustand(0.1, 99.0, self.teilchen)

        trajektorien = self.datenverwalter.hole_alle_trajektorien()

        # Sollte Trajektorien für alle Teilchen zurückgeben
        self.assertEqual(len(trajektorien), konst.N_TEILCHEN)

        # Jede Trajektorie sollte korrekte Struktur haben
        for i, (x_pos, y_pos) in enumerate(trajektorien):
            self.assertEqual(len(x_pos), 2)
            self.assertEqual(len(y_pos), 2)
            self.assertIsInstance(x_pos[0], float)
            self.assertIsInstance(y_pos[0], float)

    def test_hole_energie_historie(self):
        """Teste Abruf der Energiehistorie."""
        # Erfasse Energieentwicklung
        energien = [100.0, 99.5, 99.2, 99.0]
        zeiten = [0.0, 0.1, 0.2, 0.3]

        for t, e in zip(zeiten, energien):
            self.datenverwalter.erfasse_zustand(t, e, self.teilchen)

        t_historie, e_historie = self.datenverwalter.hole_energie_historie()

        self.assertEqual(t_historie, zeiten)
        self.assertEqual(e_historie, energien)

    def test_lade_aus_datei(self):
        """Teste Laden von Daten aus CSV-Datei."""
        # Erstelle ordnungsgemäße CSV mit allen Teilchen
        for t in range(3):
            self.datenverwalter.erfasse_zustand(t * 0.1, 100.0 - t, self.teilchen)
        self.datenverwalter.speichern()

        # Lade mit neuem Verwalter
        neuer_verwalter = Datenverwalter()
        neuer_verwalter.lade_aus_datei(self.test_datei)

        # Verifiziere geladene Daten
        self.assertEqual(len(neuer_verwalter.trajektorien_daten['zeit']), 3)
        self.assertEqual(neuer_verwalter.trajektorien_daten['zeit'][0], 0.0)
        self.assertEqual(neuer_verwalter.trajektorien_daten['energie'][2], 98.0)

        # Verifiziere, dass alle Teilchendaten geladen wurden
        self.assertEqual(
            len(neuer_verwalter.trajektorien_daten['teilchen'][0]), konst.N_TEILCHEN)

    def test_hole_statistiken(self):
        """Teste Statistikberechnung mit korrekter numerischer Präzision."""
        # Erfasse Daten mit Energieentwicklung
        energien = [100.0, 99.8, 99.7, 99.5, 99.3]
        for i, e in enumerate(energien):
            self.datenverwalter.erfasse_zustand(i * 0.1, e, self.teilchen)

        statistiken = self.datenverwalter.hole_statistiken()

        # Verifiziere Statistiken mit angemessener Präzision
        self.assertEqual(statistiken['anfangsenergie'], 100.0)
        self.assertEqual(statistiken['endenergie'], 99.3)

        # Behandle Fließkomma-Präzision korrekt
        self.assertAlmostEqual(statistiken['energie_drift'], -0.7, places=10)
        self.assertAlmostEqual(statistiken['relative_drift'], -0.007, places=10)

        self.assertEqual(statistiken['anzahl_zeitschritte'], 5)
        self.assertAlmostEqual(
            statistiken['mittlere_energie'], np.mean(energien), places=10)
        self.assertAlmostEqual(
            statistiken['std_energie'], np.std(energien), places=10)

    def test_leere_statistiken(self):
        """Teste Statistiken ohne Daten."""
        statistiken = self.datenverwalter.hole_statistiken()
        self.assertEqual(statistiken, {})

    def test_ausgabeformat_spezifikation(self):
        """Teste, dass Ausgabeformat der Projektspezifikation entspricht."""
        # Erfasse einen vollständigen Zeitschritt
        self.datenverwalter.erfasse_zustand(1.234, 567.89, self.teilchen)

        # Verifiziere Pufferformat
        zeile = self.datenverwalter.datenpuffer[0]

        # Prüfe Gesamtspaltenzahl
        erwartete_laenge = 2 + 4 * konst.N_TEILCHEN
        self.assertEqual(len(zeile), erwartete_laenge)

        # Verifiziere Zeit- und Energiespalten
        self.assertEqual(zeile[0], 1.234)
        self.assertEqual(zeile[1], 567.89)

        # Verifiziere, dass Teilchendatenspalten dem Spezifikationsformat folgen
        # Format: t, E_total, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...
        self.assertEqual(zeile[2], 10.0)   # x1
        self.assertEqual(zeile[3], 20.0)   # y1
        self.assertEqual(zeile[4], 5.0)    # vx1
        self.assertEqual(zeile[5], -3.0)   # vy1
        self.assertEqual(zeile[6], 20.0)   # x2
        self.assertEqual(zeile[7], 25.0)   # y2

    def test_datei_io_fehlerbehandlung(self):
        """Teste elegante Behandlung von Datei-I/O-Fehlern."""
        # Versuche Speichern in ungültigen Pfad
        ungueltig_verwalter = Datenverwalter("/ungueltig/pfad/datei.csv")
        ungueltig_verwalter.erfasse_zustand(0.0, 100.0, self.teilchen)

        # Sollte Fehler elegant ohne Absturz behandeln
        ungueltig_verwalter.speichern()

        # Teste Laden nicht existierender Datei
        ungueltig_verwalter.lade_aus_datei("/nichtexistent/datei.csv")

    def test_grosser_datensatz(self):
        """Teste Behandlung großer Datensätze."""
        # Erfasse viele Zeitschritte
        n_schritte = 1000
        for i in range(n_schritte):
            self.datenverwalter.erfasse_zustand(i * 0.001, 100.0, self.teilchen)

        # Verifiziere Datenstruktur
        self.assertEqual(len(self.datenverwalter.datenpuffer), n_schritte)
        self.assertEqual(self.datenverwalter.geschriebene_datensaetze, n_schritte)

        # Speichere und verifiziere
        self.datenverwalter.speichern()
        self.assertTrue(os.path.exists(self.test_datei))

    def test_entspricht_hauptsimulations_ausgabe(self):
        """Teste, dass Datenverwalter Ausgabe passend zur Hauptsimulation produziert."""
        # Verwende tatsächliche Anfangszustände aus Hauptsimulation
        teilchen = []
        for zustand in konst.ANFANGSZUSTAENDE:
            t = Teilchen(
                x=zustand[0], y=zustand[1],
                vx=zustand[2], vy=zustand[3],
                masse=konst.MASSE, ladung=konst.LADUNG
            )
            teilchen.append(t)

        # Erfasse Zustand wie Hauptsimulation es tut
        zeit = 1.234
        energie = 6658.979403  # Ähnlich der Anfangsenergie der Hauptsimulation

        self.datenverwalter.erfasse_zustand(zeit, energie, teilchen)

        # Verifiziere ordnungsgemäße Aufzeichnung
        self.assertEqual(len(self.datenverwalter.datenpuffer), 1)
        zeile = self.datenverwalter.datenpuffer[0]

        # Sollte Spezifikation exakt entsprechen
        self.assertEqual(zeile[0], zeit)
        self.assertEqual(zeile[1], energie)

        # Teilchendaten sollten Anfangszuständen entsprechen
        self.assertEqual(zeile[2], 1.0)    # x1 aus ANFANGSZUSTAENDE[0][0]
        self.assertEqual(zeile[3], 45.0)   # y1 aus ANFANGSZUSTAENDE[0][1]
        self.assertEqual(zeile[4], 10.0)   # vx1 aus ANFANGSZUSTAENDE[0][2]
        self.assertEqual(zeile[5], 0.0)    # vy1 aus ANFANGSZUSTAENDE[0][3]

    def test_trajektorien_daten_konsistenz(self):
        """Teste, dass Trajektoriendaten über Operationen hinweg konsistent bleiben."""
        # Erfasse mehrere Zeitschritte
        for t in range(10):
            # Modifiziere Teilchenpositionen leicht in jedem Schritt
            for i, teilchen in enumerate(self.teilchen):
                teilchen.zustand[0] = 10.0 + i * 10.0 + t * 0.1  # x
                teilchen.zustand[1] = 20.0 + i * 5.0 + t * 0.05  # y

            self.datenverwalter.erfasse_zustand(
                t * 0.01, 1000.0 - t, self.teilchen)

        # Verifiziere, dass alle Trajektorien-Zugriffsmethoden konsistent funktionieren
        alle_trajektorien = self.datenverwalter.hole_alle_trajektorien()

        for i in range(konst.N_TEILCHEN):
            individuelle_trajektorie = self.datenverwalter.hole_teilchen_trajektorie(i)

            # Individueller und Massenzugriff sollten gleiches Ergebnis liefern
            np.testing.assert_array_equal(
                individuelle_trajektorie[0], alle_trajektorien[i][0])
            np.testing.assert_array_equal(
                individuelle_trajektorie[1], alle_trajektorien[i][1])

    def test_energie_historie_konsistenz(self):
        """Teste Datenkonsistenz der Energiehistorie."""
        energien = [1000.0, 999.5, 999.1, 998.8, 998.5]
        zeiten = [0.0, 0.1, 0.2, 0.3, 0.4]

        for t, e in zip(zeiten, energien):
            self.datenverwalter.erfasse_zustand(t, e, self.teilchen)

        # Hole Energiehistorie
        t_hist, e_hist = self.datenverwalter.hole_energie_historie()

        # Sollte Eingabe exakt entsprechen
        self.assertEqual(len(t_hist), len(zeiten))
        self.assertEqual(len(e_hist), len(energien))

        for i in range(len(zeiten)):
            self.assertEqual(t_hist[i], zeiten[i])
            self.assertEqual(e_hist[i], energien[i])

    def test_statistik_praezision(self):
        """Teste Statistikberechnung mit korrekter Fließkomma-Behandlung."""
        # Verwende einfache Werte, die Fließkomma-Präzisionsprobleme vermeiden
        energien = [100.0, 99.0, 98.0, 97.0, 96.0]
        for i, e in enumerate(energien):
            self.datenverwalter.erfasse_zustand(i * 0.1, e, self.teilchen)

        statistiken = self.datenverwalter.hole_statistiken()

        # Verifiziere Grundstatistiken
        self.assertEqual(statistiken['anfangsenergie'], 100.0)
        self.assertEqual(statistiken['endenergie'], 96.0)
        self.assertEqual(statistiken['energie_drift'], -4.0)
        self.assertEqual(statistiken['relative_drift'], -0.04)
        self.assertEqual(statistiken['anzahl_zeitschritte'], 5)

        # Verifiziere berechnete Statistiken
        erwarteter_mittelwert = np.mean(energien)
        erwartete_std = np.std(energien)
        self.assertAlmostEqual(statistiken['mittlere_energie'], erwarteter_mittelwert, places=10)
        self.assertAlmostEqual(statistiken['std_energie'], erwartete_std, places=10)

    def test_csv_rundreise(self):
        """Teste vollständigen Speicher-/Lade-Zyklus bewahrt Daten."""
        # Erfasse ursprüngliche Daten
        urspruengliche_zeiten = []
        urspruengliche_energien = []

        for t in range(5):
            zeit_wert = t * 0.1
            energie_wert = 1000.0 - t * 0.5

            self.datenverwalter.erfasse_zustand(
                zeit_wert, energie_wert, self.teilchen)
            urspruengliche_zeiten.append(zeit_wert)
            urspruengliche_energien.append(energie_wert)

        # Speichere in Datei
        self.datenverwalter.speichern()

        # Lade mit neuem Verwalter
        neuer_verwalter = Datenverwalter()
        neuer_verwalter.lade_aus_datei(self.test_datei)

        # Vergleiche ursprüngliche und geladene Daten
        geladene_zeiten, geladene_energien = neuer_verwalter.hole_energie_historie()

        np.testing.assert_array_equal(geladene_zeiten, urspruengliche_zeiten)
        np.testing.assert_array_equal(geladene_energien, urspruengliche_energien)

        # Verifiziere Bewahrung der Teilchendaten
        for i in range(konst.N_TEILCHEN):
            orig_traj = self.datenverwalter.hole_teilchen_trajektorie(i)
            geladene_traj = neuer_verwalter.hole_teilchen_trajektorie(i)

            np.testing.assert_array_equal(orig_traj[0], geladene_traj[0])
            np.testing.assert_array_equal(orig_traj[1], geladene_traj[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)