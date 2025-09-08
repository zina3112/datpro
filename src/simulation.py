"""
Hauptsimulationscontroller für geladene Teilchen in einer Box

Orchestriert die Simulation unter Verwendung der exakten
interpolationsbasierten Kollisionserkennung.
"""

import numpy as np
import os
from typing import List, Tuple, Optional
from . import konstanten as konst
from .teilchen import Teilchen
from .box import Box
from .integrator import RK4Integrator, rk4_schritt_system
from .kraefte import (berechne_potentielle_energie_coulomb,
                      berechne_system_kraefte)
from .datenverwalter import Datenverwalter
import time as zeit_modul


class Simulation:
    """
    Hauptsimulationscontroller für geladene Teilchen in einer Box.

    Verwendet die exakte Kollisionsbehandlungsmethode:
    - Berechnet RK4-Schritte für Teilchen
    - Verwendet lineare Interpolation zur Erkennung von Wandkollisionen
    - Teilt Zeitschritte an Kollisionspunkten
    - Wendet perfekte elastische Reflexionen an
    """

    def __init__(self,
                 anfangszustaende: Optional[np.ndarray] = None,
                 dt: float = konst.DT,
                 ausgabedatei: str = None):
        """
        Initialisiert Simulation mit Teilchen und Parametern.

        Args:
            anfangszustaende: Anfangs-Zustandsvektoren [x, y, vx, vy]
            dt: Zeitschritt für Integration
            ausgabedatei: Pfad zur CSV-Ausgabedatei
        """
        if anfangszustaende is None:
            anfangszustaende = konst.ANFANGSZUSTAENDE

        # Initialisiere Teilchen aus Anfangszuständen
        self.teilchen = []
        for i, zustand in enumerate(anfangszustaende):
            teilchen = Teilchen(
                x=zustand[0], y=zustand[1],
                vx=zustand[2], vy=zustand[3],
                masse=konst.MASSE,
                ladung=konst.LADUNG
            )
            self.teilchen.append(teilchen)

        # Initialisiere Simulationskomponenten
        self.box = Box()
        self.integrator = RK4Integrator(dt=dt)
        self.datenverwalter = Datenverwalter(ausgabedatei)

        # Simulationszustand
        self.aktuelle_zeit = 0.0
        self.schrittzaehler = 0
        self.dt = dt

        # Energieverfolgung
        self.anfangsenergie = self.berechne_gesamtenergie()
        self.energie_historie = [self.anfangsenergie]
        self.zeit_historie = [0.0]

        # Erfasse Anfangszustand
        self.datenverwalter.erfasse_zustand(0.0, self.anfangsenergie, self.teilchen)

        # Leistungsverfolgung
        self.start_echtzeit = None
        self.gesamt_rechenzeit = 0.0

        print(f"Simulation initialisiert mit {len(self.teilchen)} Teilchen")
        print(f"Anfangs-Gesamtenergie: {self.anfangsenergie:.6f}")
        print(f"Zeitschritt: {self.dt}, Box: {self.box}")

    def berechne_gesamtenergie(self) -> float:
        """
        Berechnet Gesamtenergie des Systems.

        Gesamtenergie = Kinetische + Gravitationspotential + Coulombpotential

        Returns:
            Gesamte Systemenergie
        """
        gesamtenergie = 0.0

        # Summiere kinetische und gravitationelle potentielle Energie
        for teilchen in self.teilchen:
            gesamtenergie += teilchen.kinetische_energie()
            gesamtenergie += teilchen.potentielle_energie_gravitation()

        # Füge Coulomb-Potentialenergie hinzu (einmal für alle Paare)
        gesamtenergie += berechne_potentielle_energie_coulomb(self.teilchen)

        return gesamtenergie

    def schritt(self) -> bool:
        """
        Führt einen Simulationszeitschritt durch.

        Implementiert die Spezifikationsanforderung:
        1. Berechne RK4-Inkremente für alle Teilchen
        2. Prüfe für jedes Teilchen ob es Box verlassen würde
        3. Falls ja, verwende exakte Interpolationsmethode
        4. Aktualisiere alle Teilchen mit finalen Zuständen

        Returns:
            True wenn Schritt erfolgreich
        """
        # Speichere ursprüngliche Zustände für Batch-Update
        urspruengliche_zustaende = [p.zustand.copy() for p in self.teilchen]

        # Liste für finale Zustände nach Kollisionsbehandlung
        finale_zustaende = []

        # Verarbeite jedes Teilchen einzeln für Kollisionserkennung
        # Nacheinander für jedes Teilchen wie in Spezifikation
        for i, teilchen in enumerate(self.teilchen):
            # Verwende exakte Interpolationsmethode aus Box-Klasse
            # Diese Methode behandelt:
            # - Volle RK4-Berechnung
            # - Kollisionserkennung via linearer Interpolation
            # - Zeitschritt-Aufteilung am Kollisionspunkt
            # - Geschwindigkeitsreflexion und Fortsetzung

            finaler_zustand = self.box.behandle_wandkollision_exakt(
                teilchen,
                self.teilchen,
                i,
                self.dt
            )

            finale_zustaende.append(finaler_zustand)

        # Aktualisiere alle Teilchen mit finalen Zuständen
        # Statevektoren erst aktualisiert nachdem für alle Teilchen berechnet
        for teilchen, finaler_zustand in zip(self.teilchen, finale_zustaende):
            teilchen.aktualisiere_zustand(finaler_zustand)

        # Sicherheitsprüfung - stelle sicher dass alle in Grenzen
        for teilchen in self.teilchen:
            self.box.erzwinge_grenzen(teilchen)

        # Aktualisiere Simulationszeit
        self.aktuelle_zeit += self.dt
        self.schrittzaehler += 1

        # Berechne und verfolge Energie
        aktuelle_energie = self.berechne_gesamtenergie()
        self.energie_historie.append(aktuelle_energie)
        self.zeit_historie.append(self.aktuelle_zeit)

        # Prüfe Energieerhaltung
        if abs(self.anfangsenergie) > konst.EPSILON:
            energie_drift = abs(aktuelle_energie - self.anfangsenergie) / abs(self.anfangsenergie)

            # Warne wenn Drift > 1%
            if energie_drift > 0.01:
                print(f"Warnung: Energiedrift = {energie_drift * 100:.2f}% bei t={self.aktuelle_zeit:.3f}")

            # Fehler wenn Drift > 10%
            if energie_drift > 0.1:
                print(f"FEHLER: Übermäßige Energiedrift = {energie_drift * 100:.2f}%")
                print("Prüfe Zeitschrittgröße oder Kollisionsbehandlung")

        # Erfasse Daten für Ausgabe
        self.datenverwalter.erfasse_zustand(
            self.aktuelle_zeit,
            aktuelle_energie,
            self.teilchen
        )

        return True

    def schritt_alternativ_batch(self) -> bool:
        """
        Alternative Implementierung mit Batch-RK4 gefolgt von Kollisionsbehandlung.

        Kann zum Vergleich verwendet werden.
        """
        # Speichere ursprüngliche Zustände
        urspruengliche_zustaende = [p.zustand.copy() for p in self.teilchen]

        # Berechne RK4-Inkremente für alle auf einmal
        zustandsinkremente = rk4_schritt_system(self.teilchen, self.dt)

        # Wende Inkremente an und behandle Kollisionen
        for i, (teilchen, inkrement) in enumerate(zip(self.teilchen, zustandsinkremente)):
            vorlaeufiger_zustand = urspruengliche_zustaende[i] + inkrement

            # Prüfe ob Teilchen außerhalb Box
            if not self.box.ist_innerhalb(vorlaeufiger_zustand[0:2]):
                # Verwende exakte Kollisionsbehandlung
                finaler_zustand = self.box.behandle_wandkollision_exakt(
                    teilchen,
                    self.teilchen,
                    i,
                    self.dt
                )
                teilchen.aktualisiere_zustand(finaler_zustand)
            else:
                # Keine Kollision
                teilchen.aktualisiere_zustand(vorlaeufiger_zustand)

        # Aktualisiere Zeit und erfasse Daten
        self.aktuelle_zeit += self.dt
        self.schrittzaehler += 1

        aktuelle_energie = self.berechne_gesamtenergie()
        self.energie_historie.append(aktuelle_energie)
        self.zeit_historie.append(self.aktuelle_zeit)

        self.datenverwalter.erfasse_zustand(
            self.aktuelle_zeit,
            aktuelle_energie,
            self.teilchen
        )

        return True

    def laufen(self,
               simulationszeit: float = konst.SIMULATIONSZEIT,
               fortschritts_intervall: int = 1000) -> None:
        """
        Führt vollständige Simulation für angegebene Zeit aus.

        Args:
            simulationszeit: Gesamte zu simulierende Zeit in Sekunden
            fortschritts_intervall: Schritte zwischen Fortschrittsmeldungen
        """
        print(f"\nStarte Simulation für {simulationszeit} Sekunden...")
        print(f"Verwende EXAKTE Interpolationsmethode für Wandkollisionen")

        # Validiere Parameter
        if abs(self.dt) < konst.EPSILON:
            print("Fehler: Zeitschritt ist null oder zu klein")
            return

        if abs(simulationszeit) < konst.EPSILON:
            print("Simulationszeit ist null - speichere nur Anfangszustand")
            self.drucke_statistiken()
            self.datenverwalter.speichern()
            print(f"\nDaten gespeichert in {self.datenverwalter.ausgabedatei}")
            return

        # Berechne Gesamtzahl der Schritte
        gesamt_schritte = int(abs(simulationszeit / self.dt))
        print(f"Gesamtschritte: {gesamt_schritte}")
        print(f"Energietoleranz: {konst.ENERGIE_TOLERANZ}")

        # Starte Zeitmessung
        self.start_echtzeit = zeit_modul.time()

        # Hauptsimulationsschleife
        schrittzaehler = 0
        while schrittzaehler < gesamt_schritte:
            # Führe einen Zeitschritt durch
            erfolg = self.schritt()

            if not erfolg:
                print("Simulation vorzeitig aufgrund Fehler gestoppt")
                break

            schrittzaehler += 1

            # Fortschrittsmeldung
            if schrittzaehler % fortschritts_intervall == 0:
                self._drucke_fortschritt(schrittzaehler, gesamt_schritte)

        # Berechne Gesamt-Rechenzeit
        self.gesamt_rechenzeit = zeit_modul.time() - self.start_echtzeit

        # Drucke finale Statistiken
        self.drucke_statistiken()

        # Speichere Daten
        self.datenverwalter.speichern()
        print(f"\nDaten gespeichert in {self.datenverwalter.ausgabedatei}")

    def _drucke_fortschritt(self, aktueller_schritt: int, gesamt_schritte: int):
        """
        Druckt Fortschrittsinformation während Simulation.

        Args:
            aktueller_schritt: Aktuelle Schrittnummer
            gesamt_schritte: Gesamtzahl der Schritte
        """
        fortschritt = (aktueller_schritt / gesamt_schritte) * 100

        # Berechne Energiedrift
        if abs(self.anfangsenergie) > konst.EPSILON:
            energie_drift = abs(self.energie_historie[-1] - self.anfangsenergie) / abs(self.anfangsenergie)
        else:
            energie_drift = 0.0

        # Berechne Leistung
        vergangene_echtzeit = zeit_modul.time() - self.start_echtzeit
        schritte_pro_sekunde = aktueller_schritt / vergangene_echtzeit if vergangene_echtzeit > 0 else 0

        # Zähle Kollisionen
        gesamt_kollisionen = sum(p.kollisionszaehler for p in self.teilchen)

        print(f"Fortschritt: {fortschritt:.1f}% | "
              f"Zeit: {self.aktuelle_zeit:.3f}s | "
              f"Energiedrift: {energie_drift * 100:.4f}% | "
              f"Kollisionen: {gesamt_kollisionen} | "
              f"Leistung: {schritte_pro_sekunde:.0f} Schritte/s")

    def drucke_statistiken(self):
        """Druckt umfassende Simulationsstatistiken."""
        print("\n" + "=" * 60)
        print("SIMULATIONSSTATISTIKEN")
        print("=" * 60)

        # Zeitstatistiken
        print(f"\nZeitstatistiken:")
        print(f"  Simulationszeit: {self.aktuelle_zeit:.3f} Sekunden")
        print(f"  Anzahl Schritte: {self.schrittzaehler}")
        print(f"  Zeitschrittgröße: {self.dt}")
        print(f"  Rechenzeit: {self.gesamt_rechenzeit:.2f} Sekunden")

        if self.gesamt_rechenzeit > 0:
            geschwindigkeitsfaktor = abs(self.aktuelle_zeit) / self.gesamt_rechenzeit
            print(f"  Geschwindigkeitsverhältnis: {geschwindigkeitsfaktor:.2f}x Echtzeit")

        # Energieerhaltung
        finale_energie = self.energie_historie[-1] if self.energie_historie else self.anfangsenergie
        energie_drift = abs(finale_energie - self.anfangsenergie)

        if abs(self.anfangsenergie) > konst.EPSILON:
            relative_drift = energie_drift / abs(self.anfangsenergie)
        else:
            relative_drift = 0.0

        print(f"\nEnergieerhaltung:")
        print(f"  Anfangsenergie: {self.anfangsenergie:.6f}")
        print(f"  Endenergie: {finale_energie:.6f}")
        print(f"  Absolute Drift: {energie_drift:.6e}")
        print(f"  Relative Drift: {relative_drift * 100:.4f}%")

        # Prüfe ob Energie innerhalb Toleranz
        if relative_drift < konst.ENERGIE_TOLERANZ:
            print(f"  ✓ Energie erhalten innerhalb Toleranz ({konst.ENERGIE_TOLERANZ * 100:.4f}%)")
        else:
            print(f"  ✗ Energiedrift überschreitet Toleranz ({konst.ENERGIE_TOLERANZ * 100:.4f}%)")

        # Kollisionsstatistiken
        gesamt_kollisionen = sum(p.kollisionszaehler for p in self.teilchen)
        print(f"\nKollisionsstatistiken (mit EXAKTER Interpolation):")
        print(f"  Gesamte Wandkollisionen: {gesamt_kollisionen}")
        print(f"  Box-Kollisionszähler: {self.box.gesamt_kollisionen}")

        for i, teilchen in enumerate(self.teilchen):
            if teilchen.kollisionszaehler > 0:
                print(f"  Teilchen {i + 1}: {teilchen.kollisionszaehler} Kollisionen")

        # Finale Teilchenzustände
        print(f"\nFinale Teilchenzustände:")
        for i, teilchen in enumerate(self.teilchen):
            print(f"  Teilchen {i + 1}: pos=({teilchen.x:.2f}, {teilchen.y:.2f}), "
                  f"ges=({teilchen.vx:.2f}, {teilchen.vy:.2f})")

        print("=" * 60)

    def hole_trajektorie(self, teilchen_index: int) -> Tuple[List[float], List[float]]:
        """
        Holt Trajektorie eines spezifischen Teilchens.

        Args:
            teilchen_index: Index des Teilchens (0-basiert)

        Returns:
            Tupel von (x_positionen, y_positionen)
        """
        return self.datenverwalter.hole_teilchen_trajektorie(teilchen_index)

    def hole_energie_historie(self) -> Tuple[List[float], List[float]]:
        """
        Holt Energiehistorie der Simulation.

        Returns:
            Tupel von (zeiten, energien)
        """
        return self.zeit_historie, self.energie_historie
