"""
Teilchenklasse für die Simulation geladener Teilchen

Repräsentiert ein einzelnes geladenes Teilchen im 2D-Raum mit Position,
Geschwindigkeit, Masse und Ladung. Der Zustandsvektor [x, y, vx, vy]
wird für die RK4-Integration verwendet.
"""

import numpy as np
from typing import Tuple, Optional
from . import konstanten as konst


class Teilchen:
    """
    Repräsentiert ein geladenes Teilchen im 2D-Raum.

    Der Statevektor [x, y, vx, vy] stellt einen 4D Vektor dar und wird
    für das RK4 Integrationsverfahren eingeführt.

    Attribute:
        masse: Masse des Teilchens
        ladung: Ladung des Teilchens
        zustand: Zustandsvektor [x, y, vx, vy]
        teilchen_id: Eindeutige ID für jedes Teilchen
    """

    # Klassenvariable für eindeutige IDs
    _naechste_id = 0

    def __init__(self,
                 x: float,
                 y: float,
                 vx: float,
                 vy: float,
                 masse: Optional[float] = None,
                 ladung: Optional[float] = None):
        """
        Initialisiert ein Teilchen mit Position und Geschwindigkeit.

        Args:
            x: Anfangs-x-Koordinate
            y: Anfangs-y-Koordinate
            vx: Anfangsgeschwindigkeit in x-Richtung
            vy: Anfangsgeschwindigkeit in y-Richtung
            masse: Teilchenmasse (Standard aus Konstanten)
            ladung: Teilchenladung (Standard aus Konstanten)
        """
        # Eindeutige ID zuweisen
        self.teilchen_id = Teilchen._naechste_id
        Teilchen._naechste_id += 1

        # Physikalische Eigenschaften
        self.masse = masse if masse is not None else konst.MASSE
        self.ladung = ladung if ladung is not None else konst.LADUNG

        # Zustandsvektor [x, y, vx, vy] initialisieren
        # NumPy-Array für effiziente Vektoroperationen
        self.zustand = np.array([x, y, vx, vy], dtype=np.float64)

        # Anfangszustand für spätere Analyse speichern
        self.anfangszustand = self.zustand.copy()

        # Kollisionsstatistik
        self.kollisionszaehler = 0
        self.letzte_kollisionszeit = -1.0

    @property
    def position(self) -> np.ndarray:
        """
        Gibt Positionsvektor [x, y] zurück.
        """
        return self.zustand[0:2]

    @position.setter
    def position(self, pos: np.ndarray):
        """
        Setzt neue Position des Teilchens.
        """
        self.zustand[0:2] = pos

    @property
    def geschwindigkeit(self) -> np.ndarray:
        """
        Gibt Geschwindigkeitsvektor [vx, vy] zurück.
        """
        return self.zustand[2:4]

    @geschwindigkeit.setter
    def geschwindigkeit(self, vel: np.ndarray):
        """
        Setzt neue Geschwindigkeit des Teilchens.
        """
        self.zustand[2:4] = vel

    @property
    def x(self) -> float:
        """x-Koordinate des Teilchens."""
        return self.zustand[0]

    @property
    def y(self) -> float:
        """y-Koordinate des Teilchens."""
        return self.zustand[1]

    @property
    def vx(self) -> float:
        """Geschwindigkeit in x-Richtung."""
        return self.zustand[2]

    @property
    def vy(self) -> float:
        """Geschwindigkeit in y-Richtung."""
        return self.zustand[3]

    def kinetische_energie(self) -> float:
        """
        Berechnet kinetische Energie des Teilchens.

        KE = (1/2) * m * (vx^2 + vy^2)

        Returns:
            Kinetische Energie
        """
        v_quadrat = np.dot(self.geschwindigkeit, self.geschwindigkeit)
        return 0.5 * self.masse * v_quadrat

    def potentielle_energie_gravitation(self) -> float:
        """
        Berechnet gravitationelle potentielle Energie.

        PE_grav = -m * g * y

        Returns:
            Gravitationelle potentielle Energie
        """
        return -self.masse * konst.GRAVITATION * self.y

    def abstand_zu(self, anderes: 'Teilchen') -> float:
        """
        Berechnet Abstand zu einem anderen Teilchen.

        Args:
            anderes: Anderes Teilchen

        Returns:
            Euklidischer Abstand
        """
        verschiebung = self.position - anderes.position
        return np.linalg.norm(verschiebung)

    def verschiebung_zu(self, anderes: 'Teilchen') -> np.ndarray:
        """
        Berechnet Verschiebungsvektor zu anderem Teilchen.

        Vektor zeigt von anderem Teilchen zu diesem.

        Args:
            anderes: Anderes Teilchen

        Returns:
            Verschiebungsvektor [dx, dy]
        """
        return self.position - anderes.position

    def aktualisiere_zustand(self, neuer_zustand: np.ndarray):
        """
        Aktualisiert Teilchenzustand mit neuem Zustandsvektor.

        Args:
            neuer_zustand: Neuer Zustandsvektor [x, y, vx, vy]
        """
        if len(neuer_zustand) != 4:
            raise ValueError(f"Zustandsvektor muss 4 Komponenten haben, hat {len(neuer_zustand)}")

        # Zustand aktualisieren (Kopie um Referenzprobleme zu vermeiden)
        self.zustand = np.array(neuer_zustand, dtype=np.float64)

    def kopiere(self) -> 'Teilchen':
        """
        Erstellt tiefe Kopie des Teilchens.

        Returns:
            Neues Teilchen mit gleichen Eigenschaften
        """
        neues_teilchen = Teilchen(
            self.x, self.y, self.vx, self.vy,
            masse=self.masse, ladung=self.ladung
        )
        neues_teilchen.kollisionszaehler = self.kollisionszaehler
        neues_teilchen.letzte_kollisionszeit = self.letzte_kollisionszeit
        neues_teilchen.anfangszustand = self.anfangszustand.copy()
        return neues_teilchen

    def __str__(self) -> str:
        """String-Darstellung des Teilchens."""
        return (f"Teilchen {self.teilchen_id}: "
                f"pos=({self.x:.2f}, {self.y:.2f}), "
                f"ges=({self.vx:.2f}, {self.vy:.2f}), "
                f"m={self.masse}, q={self.ladung}")

    def __repr__(self) -> str:
        """Technische Darstellung für Debugging."""
        return (f"Teilchen(x={self.x}, y={self.y}, "
                f"vx={self.vx}, vy={self.vy}, "
                f"masse={self.masse}, ladung={self.ladung}")
