"""
Runge-Kutta 4. Ordnung (RK4) numerischer Integrator

Implementiert die RK4-Methode zur Lösung der Bewegungsdifferentialgleichungen.
Die DGL zweiter Ordnung wird durch Einführung des Zustandsvektors
in ein System erster Ordnung umgewandelt.

Der Zustandsvektor ist [x, y, vx, vy] und die Ableitung ist [vx, vy, ax, ay].
"""

import numpy as np
from typing import Callable, List
from . import konstanten as konst
from .teilchen import Teilchen
from .kraefte import berechne_system_beschleunigungen


def zustandsableitung(teilchen: List[Teilchen]) -> List[np.ndarray]:
    """
    Berechnet die Ableitung des Zustandsvektors für alle Teilchen.

    Für Zustandsvektor s = [x, y, vx, vy] ist die Ableitung:
    ds/dt = [vx, vy, ax, ay]

    wobei die Beschleunigungen aus den Kräften berechnet werden.

    Args:
        teilchen: Liste aller Teilchen

    Returns:
        Liste der Ableitungsvektoren für jedes Teilchen
    """
    # Berechne Beschleunigungen für alle Teilchen
    beschleunigungen = berechne_system_beschleunigungen(teilchen)

    ableitungen = []
    for aktuelles_teilchen, beschleunigung in zip(teilchen, beschleunigungen):
        # Extrahiere Geschwindigkeit aus Zustand
        geschwindigkeit = aktuelles_teilchen.geschwindigkeit

        # Konstruiere Ableitung: [vx, vy, ax, ay]
        ableitung = np.zeros(4)
        ableitung[0:2] = geschwindigkeit  # dx/dt = vx, dy/dt = vy
        ableitung[2:4] = beschleunigung  # dvx/dt = ax, dvy/dt = ay

        ableitungen.append(ableitung)

    return ableitungen


def rk4_schritt_einzeln(teilchen: Teilchen,
                        alle_teilchen: List[Teilchen],
                        teilchen_index: int,
                        dt: float) -> np.ndarray:
    """
    Führt einen RK4-Integrationsschritt für ein einzelnes Teilchen durch.

    Diese Funktion berechnet das Zustandsinkrement für ein Teilchen
    unter Berücksichtigung der Kräfte von allen anderen Teilchen.
    Wichtig für Kollisionsbehandlung wo Trajektorie nach Kollision
    neu berechnet werden muss.

    Args:
        teilchen: Das zu integrierende Teilchen
        alle_teilchen: Liste aller Teilchen (für Kraftberechnungen)
        teilchen_index: Index des Teilchens
        dt: Zeitschrittgröße

    Returns:
        Zustandsinkrement [Δx, Δy, Δvx, Δvy]
    """
    # Behandle Null-Zeitschritt
    if abs(dt) < 1e-15:
        return np.zeros(4)

    # Speichere ursprüngliche Zustände aller Teilchen
    urspruengliche_zustaende = [p.zustand.copy() for p in alle_teilchen]

    try:
        # Berechne k1 = dt * f(s_n)
        ableitungen_k1 = zustandsableitung(alle_teilchen)
        k1 = dt * ableitungen_k1[teilchen_index]

        # Berechne k2 mit Zustand bei s_n + k1/2
        # Alle Teilchen müssen für konsistente Kraftberechnung verschoben werden
        for i, p in enumerate(alle_teilchen):
            if i == teilchen_index:
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * k1)
            else:
                # Andere Teilchen bewegen sich mit ihrem eigenen k1
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * dt * ableitungen_k1[i])

        ableitungen_k2 = zustandsableitung(alle_teilchen)
        k2 = dt * ableitungen_k2[teilchen_index]

        # Berechne k3 mit Zustand bei s_n + k2/2
        for i, p in enumerate(alle_teilchen):
            if i == teilchen_index:
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * k2)
            else:
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * dt * ableitungen_k2[i])

        ableitungen_k3 = zustandsableitung(alle_teilchen)
        k3 = dt * ableitungen_k3[teilchen_index]

        # Berechne k4 mit Zustand bei s_n + k3
        for i, p in enumerate(alle_teilchen):
            if i == teilchen_index:
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + k3)
            else:
                p.aktualisiere_zustand(urspruengliche_zustaende[i] + dt * ableitungen_k3[i])

        ableitungen_k4 = zustandsableitung(alle_teilchen)
        k4 = dt * ableitungen_k4[teilchen_index]

        # Kombiniere RK4-Koeffizienten
        # Inkrement = (k1 + 2*k2 + 2*k3 + k4) / 6
        zustandsinkrement = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    finally:
        # Stelle immer ursprüngliche Zustände wieder her
        for i, p in enumerate(alle_teilchen):
            p.aktualisiere_zustand(urspruengliche_zustaende[i])

    return zustandsinkrement


def rk4_schritt_system(teilchen: List[Teilchen], dt: float) -> List[np.ndarray]:
    """
    Führt einen RK4-Integrationsschritt für das gesamte Teilchensystem durch.

    Implementiert Batch-Update: Alle Kräfte werden berechnet bevor
    irgendwelche Zustände aktualisiert werden. Standard-Ansatz für
    gekoppelte Systeme von ODEs.

    Args:
        teilchen: Liste aller Teilchen
        dt: Zeitschrittgröße

    Returns:
        Liste der Zustandsinkremente für alle Teilchen
    """
    # Behandle Null-Zeitschritt
    if abs(dt) < 1e-15:
        return [np.zeros(4) for _ in teilchen]

    n_teilchen = len(teilchen)

    # Speichere ursprüngliche Zustände
    urspruengliche_zustaende = [p.zustand.copy() for p in teilchen]

    # Speicher für RK4-Koeffizienten
    k1_liste = []
    k2_liste = []
    k3_liste = []
    k4_liste = []

    try:
        # Berechne k1 für alle Teilchen
        # k1 = dt * f(s_n)
        ableitungen = zustandsableitung(teilchen)
        for ableitung in ableitungen:
            k1_liste.append(dt * ableitung)

        # Berechne k2 für alle Teilchen
        # Bewege alle zu s_n + k1/2
        for i, aktuelles_teilchen in enumerate(teilchen):
            aktuelles_teilchen.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * k1_liste[i])

        # Berechne Ableitungen am Mittelpunkt
        ableitungen = zustandsableitung(teilchen)
        for ableitung in ableitungen:
            k2_liste.append(dt * ableitung)

        # Berechne k3 für alle Teilchen
        # Bewege alle zu s_n + k2/2
        for i, aktuelles_teilchen in enumerate(teilchen):
            aktuelles_teilchen.aktualisiere_zustand(urspruengliche_zustaende[i] + 0.5 * k2_liste[i])

        ableitungen = zustandsableitung(teilchen)
        for ableitung in ableitungen:
            k3_liste.append(dt * ableitung)

        # Berechne k4 für alle Teilchen
        # Bewege alle zu s_n + k3
        for i, aktuelles_teilchen in enumerate(teilchen):
            aktuelles_teilchen.aktualisiere_zustand(urspruengliche_zustaende[i] + k3_liste[i])

        ableitungen = zustandsableitung(teilchen)
        for ableitung in ableitungen:
            k4_liste.append(dt * ableitung)

        # Kombiniere RK4-Koeffizienten für jedes Teilchen
        zustandsinkremente = []
        for i in range(n_teilchen):
            inkrement = (k1_liste[i] + 2 * k2_liste[i] + 2 * k3_liste[i] + k4_liste[i]) / 6.0
            zustandsinkremente.append(inkrement)

    finally:
        # Stelle immer ursprüngliche Zustände wieder her
        for i, aktuelles_teilchen in enumerate(teilchen):
            aktuelles_teilchen.aktualisiere_zustand(urspruengliche_zustaende[i])

    return zustandsinkremente


class RK4Integrator:
    """
    Runge-Kutta 4. Ordnung Integrator für Teilchensystem.

    Kapselt die RK4-Integrationsmethode und bietet saubere
    Schnittstelle für Zeitintegration des Teilchensystems.
    """

    def __init__(self, dt: float = konst.DT):
        """
        Initialisiert den RK4-Integrator.

        Args:
            dt: Zeitschrittgröße
        """
        self.dt = dt
        self.schrittzaehler = 0
        self.gesamtzeit = 0.0

    def schritt(self, teilchen: List[Teilchen]) -> List[np.ndarray]:
        """
        Bewegt Teilchensystem um einen Zeitschritt vorwärts.

        Args:
            teilchen: Liste der zu integrierenden Teilchen

        Returns:
            Zustandsinkremente für alle Teilchen
        """
        # Berechne Zustandsinkremente mit RK4
        inkremente = rk4_schritt_system(teilchen, self.dt)

        # Aktualisiere Statistiken
        self.schrittzaehler += 1
        self.gesamtzeit += self.dt

        return inkremente

    def integriere_bis_zeit(self,
                            teilchen: List[Teilchen],
                            zielzeit: float,
                            callback: Callable = None) -> List[Teilchen]:
        """
        Integriert System bis zu einer Zielzeit.

        Args:
            teilchen: Liste der Teilchen
            zielzeit: Zeit bis zu der integriert werden soll
            callback: Optionale Funktion die nach jedem Schritt aufgerufen wird

        Returns:
            Aktualisierte Teilchenliste
        """
        while abs(self.gesamtzeit - zielzeit) > konst.EPSILON:
            # Berechne verbleibende Zeit
            verbleibende_zeit = zielzeit - self.gesamtzeit
            dt_schritt = min(abs(self.dt), abs(verbleibende_zeit)) * np.sign(verbleibende_zeit)

            # Führe Integrationsschritt durch
            inkremente = rk4_schritt_system(teilchen, dt_schritt)

            # Aktualisiere Teilchenzustände
            for aktuelles_teilchen, inkrement in zip(teilchen, inkremente):
                aktuelles_teilchen.aktualisiere_zustand(aktuelles_teilchen.zustand + inkrement)

            # Aktualisiere Zeit
            self.gesamtzeit += dt_schritt
            self.schrittzaehler += 1

            # Rufe Callback auf falls vorhanden
            if callback is not None:
                callback(self.gesamtzeit, teilchen)

        return teilchen

    def zuruecksetzen(self):
        """Setzt Integratorstatistiken zurück."""
        self.schrittzaehler = 0
        self.gesamtzeit = 0.0
