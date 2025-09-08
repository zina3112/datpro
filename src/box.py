"""
Simulationsbox mit reflektierenden Wänden für geladene Teilchen

Implementiert die exakte Kollisionserkennung mittels linearer Interpolation
und perfekt elastische Reflexionen an den Boxwänden.
"""

import numpy as np
from typing import Tuple, Optional, List
from . import konstanten as konst
from .teilchen import Teilchen


class Box:
    """
    Repräsentiert die 2D-Box mit perfekt reflektierenden Wänden.

    Implementierung nach folgenden Schritten (wie im Projekt angegeben):
    1. Berechnen von rk4_step(dgl, s, dt)
    2. Prüfen, ob neuer Statevektor außerhalb Box
    3. Lineare Interpolation für Berechnung von dt (wann Teilchen die Boxwand durchquert)
    4. Durchführung von rk4 für den Bruchteil von dt vor Kollision mit Wand
    5. Geschwindigkeitsvektor reflektieren (senkrecht zur Wand)
    6. Mit reflektiertem Geschwindigkeitsvektor Rest des Zeitschrittes durchführen
    """

    def __init__(self,
                 x_min: float = konst.BOX_MIN_X,
                 x_max: float = konst.BOX_MAX_X,
                 y_min: float = konst.BOX_MIN_Y,
                 y_max: float = konst.BOX_MAX_Y):
        """Initialisiert die Box mit den gegebenen Grenzen."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.breite = x_max - x_min
        self.hoehe = y_max - y_min

        if self.breite <= 0 or self.hoehe <= 0:
            raise ValueError("Boxdimensionen müssen positiv sein")

        # Statistiken tracken
        self.gesamt_kollisionen = 0
        self.kollisions_historie = []

    def ist_innerhalb(self, position: np.ndarray) -> bool:
        """
        Prüft, ob Position innerhalb der Boxdimension liegt.

        Args:
            position: [x, y] Positionsvektor

        Returns:
            bool: True wenn Position innerhalb Box (einschließlich Grenzen)
        """
        x, y = position
        return (self.x_min <= x <= self.x_max and
                self.y_min <= y <= self.y_max)

    def behandle_wandkollision_exakt(self,
                                     teilchen: Teilchen,
                                     alle_teilchen: List[Teilchen],
                                     teilchen_index: int,
                                     dt: float) -> np.ndarray:
        """
        Behandelt Wandkollisionen mit der exakten Interpolationsmethode.

        Diese Methode implementiert den im Projekt spezifizierten Algorithmus:
        - Berechnet zunächst den vollen RK4-Schritt
        - Prüft ob das Teilchen die Box verlassen würde
        - Verwendet lineare Interpolation zur Bestimmung der Kollisionszeit
        - Teilt den Zeitschritt am Kollisionspunkt
        - Reflektiert die Geschwindigkeit und führt Rest des Zeitschritts aus

        Args:
            teilchen: Das zu aktualisierende Teilchen
            alle_teilchen: Alle Teilchen für Kraftberechnungen
            teilchen_index: Index dieses Teilchens
            dt: Zeitschritt

        Returns:
            np.ndarray: Finaler Zustand nach Kollisionsbehandlung
        """
        from .integrator import rk4_schritt_einzeln

        # Speichere ursprünglichen Zustand vor Berechnungen
        urspruenglicher_zustand = teilchen.zustand.copy()

        # Schritt 1: Berechne vollen RK4-Schritt als ob es keine Wände gäbe
        voller_schritt_inkrement = rk4_schritt_einzeln(
            teilchen,
            alle_teilchen,
            teilchen_index,
            dt
        )

        # Berechne was der neue Zustand nach vollem Zeitschritt wäre
        vorlaeufiger_neuer_zustand = urspruenglicher_zustand + voller_schritt_inkrement

        # Schritt 2: Prüfe ob neue Position außerhalb Box liegt
        vorlaeufige_position = vorlaeufiger_neuer_zustand[0:2]

        if self.ist_innerhalb(vorlaeufige_position):
            # Keine Kollision - gib vollen Schritt zurück
            return vorlaeufiger_neuer_zustand

        # Schritt 3: Bruchteil von dt finden, nach dem das Teilchen die Boxwand durchquert
        # Verwende lineare Interpolation zwischen Start- und Endposition

        x0, y0 = urspruenglicher_zustand[0:2]  # Startposition
        x1, y1 = vorlaeufiger_neuer_zustand[0:2]  # Endposition nach dt

        # Finde kleinsten positiven Bruchteil für erste Kollision
        kollisions_bruchteil = 1.0
        getroffene_wand = None

        # Prüfe Kollision mit linker Wand (x = x_min)
        if x1 < self.x_min and x0 >= self.x_min:
            # Lineare Interpolation: finde t sodass x(t) = x_min
            # x(t) = x0 + t*(x1-x0) = x_min
            # Daraus folgt: t = (x_min - x0)/(x1 - x0)
            if abs(x1 - x0) > konst.EPSILON:  # Division durch Null vermeiden
                t_links = (self.x_min - x0) / (x1 - x0)
                if 0 <= t_links < kollisions_bruchteil:
                    kollisions_bruchteil = t_links
                    getroffene_wand = 'links'

        # Prüfe Kollision mit rechter Wand (x = x_max)
        elif x1 > self.x_max and x0 <= self.x_max:
            if abs(x1 - x0) > konst.EPSILON:
                t_rechts = (self.x_max - x0) / (x1 - x0)
                if 0 <= t_rechts < kollisions_bruchteil:
                    kollisions_bruchteil = t_rechts
                    getroffene_wand = 'rechts'

        # Prüfe Kollision mit unterer Wand (y = y_min)
        if y1 < self.y_min and y0 >= self.y_min:
            if abs(y1 - y0) > konst.EPSILON:
                t_unten = (self.y_min - y0) / (y1 - y0)
                if 0 <= t_unten < kollisions_bruchteil:
                    kollisions_bruchteil = t_unten
                    getroffene_wand = 'unten'

        # Prüfe Kollision mit oberer Wand (y = y_max)
        elif y1 > self.y_max and y0 <= self.y_max:
            if abs(y1 - y0) > konst.EPSILON:
                t_oben = (self.y_max - y0) / (y1 - y0)
                if 0 <= t_oben < kollisions_bruchteil:
                    kollisions_bruchteil = t_oben
                    getroffene_wand = 'oben'

        # Schritt 4: RK4 für Bruchteil dt vor Kollision durchführen
        # Bringt das Teilchen exakt auf die Wand
        dt_bis_kollision = kollisions_bruchteil * dt

        if dt_bis_kollision > konst.EPSILON:
            # Berechne RK4-Schritt bis zum Kollisionspunkt
            schritt_bis_kollision = rk4_schritt_einzeln(
                teilchen,
                alle_teilchen,
                teilchen_index,
                dt_bis_kollision
            )
            zustand_bei_kollision = urspruenglicher_zustand + schritt_bis_kollision
        else:
            # Kollision passiert sofort
            zustand_bei_kollision = urspruenglicher_zustand.copy()

        # Schritt 5: Geschwindigkeitskomponente senkrecht zur Wand reflektieren
        # Perfekt elastische Reflexion: Einfallswinkel = Ausfallswinkel
        reflektierter_zustand = zustand_bei_kollision.copy()

        if getroffene_wand == 'links' or getroffene_wand == 'rechts':
            # Vorzeichen der x-Geschwindigkeit umkehren
            reflektierter_zustand[2] = -reflektierter_zustand[2]
            # Kollision für Statistik erfassen
            self.gesamt_kollisionen += 1
            teilchen.kollisionszaehler += 1

        elif getroffene_wand == 'unten' or getroffene_wand == 'oben':
            # Vorzeichen der y-Geschwindigkeit umkehren
            reflektierter_zustand[3] = -reflektierter_zustand[3]
            # Kollision für Statistik erfassen
            self.gesamt_kollisionen += 1
            teilchen.kollisionszaehler += 1

        # Schritt 6: Mit reflektiertem Geschwindigkeitsvektor Rest des Zeitschritts durchführen
        dt_nach_kollision = dt - dt_bis_kollision

        if dt_nach_kollision > konst.EPSILON:
            # Temporär Teilchenzustand auf reflektierten Zustand setzen
            teilchen.aktualisiere_zustand(reflektierter_zustand)

            # RK4-Schritt für verbleibende Zeit berechnen
            schritt_nach_kollision = rk4_schritt_einzeln(
                teilchen,
                alle_teilchen,
                teilchen_index,
                dt_nach_kollision
            )

            # Ursprünglichen Zustand wiederherstellen für Batch-Updates
            teilchen.aktualisiere_zustand(urspruenglicher_zustand)

            # Finaler Zustand nach vollständigem Zeitschritt
            finaler_zustand = reflektierter_zustand + schritt_nach_kollision
        else:
            finaler_zustand = reflektierter_zustand

        # Sicherstellen dass Teilchen innerhalb Box liegt (numerische Fehler behandeln)
        if not self.ist_innerhalb(finaler_zustand[0:2]):
            finaler_zustand[0] = np.clip(
                finaler_zustand[0], self.x_min, self.x_max)
            finaler_zustand[1] = np.clip(
                finaler_zustand[1], self.y_min, self.y_max)

        # Prüfe auf Sekundärkollision (kann bei Ecken auftreten)
        if not self.ist_innerhalb(finaler_zustand[0:2]) or self._wuerde_im_naechsten_schritt_austreten(finaler_zustand, dt * 0.1):
            # Rekursiv Sekundärkollision behandeln
            if dt_nach_kollision > konst.EPSILON and self.gesamt_kollisionen < konst.MAX_KOLLISIONS_ITERATIONEN:
                teilchen.aktualisiere_zustand(reflektierter_zustand)
                finaler_zustand = self.behandle_wandkollision_exakt(
                    teilchen,
                    alle_teilchen,
                    teilchen_index,
                    dt_nach_kollision
                )
                # Ursprünglichen Zustand wiederherstellen
                teilchen.aktualisiere_zustand(urspruenglicher_zustand)

        return finaler_zustand

    def _wuerde_im_naechsten_schritt_austreten(self, zustand: np.ndarray, dt: float) -> bool:
        """
        Hilfsmethode um zu prüfen ob Teilchen im nächsten Schritt Box verlassen würde.

        Verwendet für Erkennung von Eckkollisionen.
        """
        # Einfache lineare Projektion
        naechstes_x = zustand[0] + zustand[2] * dt
        naechstes_y = zustand[1] + zustand[3] * dt

        return not self.ist_innerhalb(np.array([naechstes_x, naechstes_y]))

    def pruefe_und_behandle_kollisionen_einfach(self, teilchen: Teilchen, dt: float) -> bool:
        """
        Einfache Kollisionsbehandlung als Backup-Methode.

        Weniger genau als Interpolationsmethode aber schneller.
        Kann nach Hauptintegration für Sicherheitsprüfung verwendet werden.
        """
        kollision_aufgetreten = False

        # Horizontale Grenzen prüfen
        if teilchen.x < self.x_min:
            teilchen.zustand[0] = 2 * self.x_min - \
                teilchen.x  # Position reflektieren
            # x-Geschwindigkeit umkehren
            teilchen.zustand[2] = -teilchen.zustand[2]
            kollision_aufgetreten = True

        elif teilchen.x > self.x_max:
            teilchen.zustand[0] = 2 * self.x_max - teilchen.x
            teilchen.zustand[2] = -teilchen.zustand[2]
            kollision_aufgetreten = True

        # Vertikale Grenzen prüfen
        if teilchen.y < self.y_min:
            teilchen.zustand[1] = 2 * self.y_min - teilchen.y
            # y-Geschwindigkeit umkehren
            teilchen.zustand[3] = -teilchen.zustand[3]
            kollision_aufgetreten = True

        elif teilchen.y > self.y_max:
            teilchen.zustand[1] = 2 * self.y_max - teilchen.y
            teilchen.zustand[3] = -teilchen.zustand[3]
            kollision_aufgetreten = True

        if kollision_aufgetreten:
            self.gesamt_kollisionen += 1
            teilchen.kollisionszaehler += 1
            teilchen.letzte_kollisionszeit = dt

        return kollision_aufgetreten

    def erzwinge_grenzen(self, teilchen: Teilchen):
        """
        Stellt sicher dass Teilchen innerhalb Box-Grenzen bleibt.

        Finale Sicherheitsprüfung um numerische Fehler zu behandeln.
        """
        teilchen.zustand[0] = np.clip(
            teilchen.zustand[0], self.x_min, self.x_max)
        teilchen.zustand[1] = np.clip(
            teilchen.zustand[1], self.y_min, self.y_max)

    def __str__(self) -> str:
        """String-Darstellung der Box."""
        return (f"Box: [{self.x_min}, {self.x_max}] × [{self.y_min}, {self.y_max}], "
                f"Gesamtkollisionen: {self.gesamt_kollisionen}")
