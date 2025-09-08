"""
Datenausgabe und -verwaltungsmodul

Verwaltet alle Datenaufzeichnungen und Datei-I/O für die Simulation,
einschließlich CSV-Ausgabe und Trajektorienverwaltung.
"""

import numpy as np
import os
from typing import List, Tuple, Optional
import csv
from . import konstanten as konst
from .teilchen import Teilchen


class Datenverwalter:
    """
    Verwaltet Datenerfassung und -ausgabe für die Simulation.

    Diese Klasse behandelt:
    - Aufzeichnung von Teilchenzuständen zu jedem Zeitpunkt
    - Schreiben von Daten in CSV-Dateien
    - Verwaltung der Trajektorienhistorie
    - Bereitstellung des Datenzugriffs für die Visualisierung
    """

    def __init__(self, ausgabedatei: Optional[str] = None):
        """
        Initialisiert den Datenverwalter.

        Args:
            ausgabedatei: Pfad zur Ausgabe-CSV-Datei
        """
        if ausgabedatei is None:
            # Ausgabeverzeichnis erstellen falls es nicht existiert
            os.makedirs(konst.AUSGABE_VERZEICHNIS, exist_ok=True)
            ausgabedatei = os.path.join(konst.AUSGABE_VERZEICHNIS, konst.AUSGABE_DATEI)

        self.ausgabedatei = ausgabedatei

        # Datenspeicher initialisieren
        self.datenpuffer = []
        self.trajektorien_daten = {
            'zeit': [],
            'energie': [],
            'teilchen': []
        }

        # CSV-Header generieren
        self.header = self._generiere_header()

        # Statistiken
        self.geschriebene_datensaetze = 0

        print(f"Datenverwalter initialisiert. Ausgabedatei: {self.ausgabedatei}")

    def _generiere_header(self) -> List[str]:
        """
        Generiert CSV-Header basierend auf Anzahl der Teilchen.

        Format: t, E_total, x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...

        Returns:
            Liste der Header-Spaltennamen
        """
        header = ['t', 'E_total']

        for i in range(konst.N_TEILCHEN):
            teilchen_nr = i + 1
            header.extend([
                f'x{teilchen_nr}',
                f'y{teilchen_nr}',
                f'vx{teilchen_nr}',
                f'vy{teilchen_nr}'
            ])

        return header

    def erfasse_zustand(self,
                        zeit: float,
                        gesamtenergie: float,
                        teilchen: List[Teilchen]) -> None:
        """
        Erfasst aktuellen Zustand der Simulation.

        Args:
            zeit: Aktuelle Simulationszeit
            gesamtenergie: Gesamte Systemenergie
            teilchen: Liste aller Teilchen
        """
        # Erstelle Datenzeile
        zeile = [zeit, gesamtenergie]

        # Füge Teilchenzustände hinzu
        for aktuelles_teilchen in teilchen:
            zeile.extend([
                aktuelles_teilchen.x,
                aktuelles_teilchen.y,
                aktuelles_teilchen.vx,
                aktuelles_teilchen.vy
            ])

        # Füge zum Puffer hinzu
        self.datenpuffer.append(zeile)

        # Speichere in Trajektoriendaten für einfachen Zugriff
        self.trajektorien_daten['zeit'].append(zeit)
        self.trajektorien_daten['energie'].append(gesamtenergie)

        # Speichere Teilchenzustände
        teilchen_zustaende = []
        for aktuelles_teilchen in teilchen:
            teilchen_zustaende.append({
                'x': aktuelles_teilchen.x,
                'y': aktuelles_teilchen.y,
                'vx': aktuelles_teilchen.vx,
                'vy': aktuelles_teilchen.vy
            })
        self.trajektorien_daten['teilchen'].append(teilchen_zustaende)

        self.geschriebene_datensaetze += 1

    def speichern(self, dateiname: Optional[str] = None) -> None:
        """
        Speichert alle aufgezeichneten Daten in CSV-Datei.

        Args:
            dateiname: Optionaler alternativer Dateiname
        """
        ausgabe_datei = dateiname if dateiname else self.ausgabedatei

        try:
            with open(ausgabe_datei, 'w', newline='') as csvdatei:
                schreiber = csv.writer(csvdatei)

                # Schreibe Header
                schreiber.writerow(self.header)

                # Schreibe alle Datenzeilen
                schreiber.writerows(self.datenpuffer)

            print(f"Erfolgreich {self.geschriebene_datensaetze} Datensätze in {ausgabe_datei} gespeichert")

        except IOError as e:
            print(f"Fehler beim Speichern der Daten: {e}")

    def speichern_inkrementell(self) -> None:
        """
        Speichert Daten inkrementell (Anhängemodus).

        Nützlich für lange Simulationen zur Vermeidung von Datenverlust.
        """
        # Prüfe ob Datei existiert um zu bestimmen ob Header benötigt wird
        schreibe_header = not os.path.exists(self.ausgabedatei)

        try:
            with open(self.ausgabedatei, 'a', newline='') as csvdatei:
                schreiber = csv.writer(csvdatei)

                # Schreibe Header falls neue Datei
                if schreibe_header:
                    schreiber.writerow(self.header)

                # Schreibe gepufferte Daten
                schreiber.writerows(self.datenpuffer)

            # Lösche Puffer nach dem Schreiben
            self.datenpuffer = []

        except IOError as e:
            print(f"Fehler beim inkrementellen Speichern: {e}")

    def hole_teilchen_trajektorie(self, teilchen_index: int) -> Tuple[List[float], List[float]]:
        """
        Holt Trajektorie eines spezifischen Teilchens.

        Args:
            teilchen_index: Index des Teilchens (0-basiert)

        Returns:
            Tupel von (x_positionen, y_positionen)
        """
        if teilchen_index >= konst.N_TEILCHEN:
            raise ValueError(f"Teilchenindex {teilchen_index} außerhalb des Bereichs")

        x_positionen = []
        y_positionen = []

        for zeitschritt_daten in self.trajektorien_daten['teilchen']:
            teilchen_zustand = zeitschritt_daten[teilchen_index]
            x_positionen.append(teilchen_zustand['x'])
            y_positionen.append(teilchen_zustand['y'])

        return x_positionen, y_positionen

    def hole_alle_trajektorien(self) -> List[Tuple[List[float], List[float]]]:
        """
        Holt Trajektorien für alle Teilchen.

        Returns:
            Liste von (x_positionen, y_positionen) für jedes Teilchen
        """
        trajektorien = []

        for i in range(konst.N_TEILCHEN):
            trajektorie = self.hole_teilchen_trajektorie(i)
            trajektorien.append(trajektorie)

        return trajektorien

    def hole_energie_historie(self) -> Tuple[List[float], List[float]]:
        """
        Holt Energiehistorie.

        Returns:
            Tupel von (zeiten, energien)
        """
        return self.trajektorien_daten['zeit'], self.trajektorien_daten['energie']

    def lade_aus_datei(self, dateiname: str) -> None:
        """
        Lädt Simulationsdaten aus CSV-Datei.

        Args:
            dateiname: Pfad zur CSV-Datei
        """
        self.trajektorien_daten = {
            'zeit': [],
            'energie': [],
            'teilchen': []
        }

        try:
            with open(dateiname, 'r') as csvdatei:
                leser = csv.DictReader(csvdatei)

                for zeile in leser:
                    # Extrahiere Zeit und Energie
                    self.trajektorien_daten['zeit'].append(float(zeile['t']))
                    self.trajektorien_daten['energie'].append(float(zeile['E_total']))

                    # Extrahiere Teilchenzustände
                    teilchen_zustaende = []
                    for i in range(konst.N_TEILCHEN):
                        p_nr = i + 1
                        zustand = {
                            'x': float(zeile[f'x{p_nr}']),
                            'y': float(zeile[f'y{p_nr}']),
                            'vx': float(zeile[f'vx{p_nr}']),
                            'vy': float(zeile[f'vy{p_nr}'])
                        }
                        teilchen_zustaende.append(zustand)

                    self.trajektorien_daten['teilchen'].append(teilchen_zustaende)

            print(f"Geladen: {len(self.trajektorien_daten['zeit'])} Zeitschritte aus {dateiname}")

        except IOError as e:
            print(f"Fehler beim Laden der Daten: {e}")

    def hole_statistiken(self) -> dict:
        """
        Berechnet Statistiken aus aufgezeichneten Daten.

        Returns:
            Dictionary mit Statistiken einschließlich Energiedrift
        """
        if not self.trajektorien_daten['energie']:
            return {}

        energien = np.array(self.trajektorien_daten['energie'])

        statistiken = {
            'anfangsenergie': energien[0],
            'endenergie': energien[-1],
            'mittlere_energie': np.mean(energien),
            'std_energie': np.std(energien),
            'max_energie': np.max(energien),
            'min_energie': np.min(energien),
            'energie_drift': energien[-1] - energien[0],
            'relative_drift': (energien[-1] - energien[0]) / abs(energien[0]),
            'anzahl_zeitschritte': len(energien)
        }

        return statistiken
