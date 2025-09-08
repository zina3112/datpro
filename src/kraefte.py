"""
Kraftberechnungsmodul für geladene Teilchen

Behandelt alle Kraftberechnungen einschließlich Gravitation
und elektrostatischer (Coulomb) Abstoßung zwischen Teilchen.
"""

import numpy as np
from typing import List
from . import konstanten as konst
from .teilchen import Teilchen


def berechne_gravitationskraft(teilchen: Teilchen) -> np.ndarray:
    """
    Berechnet Gravitationskraft auf ein Teilchen.

    F_gravity = [0, m * g] wobei g negativ ist (nach unten)

    Args:
        teilchen: Teilchen für das die Kraft berechnet wird

    Returns:
        Kraftvektor [Fx, Fy]
    """
    kraft_x = 0.0
    kraft_y = teilchen.masse * konst.GRAVITATION
    return np.array([kraft_x, kraft_y])


def berechne_coulombkraft_zwischen(teilchen1: Teilchen,
                                   teilchen2: Teilchen) -> np.ndarray:
    """
    Berechnet elektrostatische Kraft zwischen zwei geladenen Teilchen.

    Verwendet Soft-Core-Regularisierung um Singularität bei r=0 zu vermeiden.
    Die Coulomb-Kraft folgt: F = q1*q2*r_ij/|r_ij|^3

    Args:
        teilchen1: Erstes Teilchen
        teilchen2: Zweites Teilchen

    Returns:
        Kraftvektor auf teilchen1 durch teilchen2
    """
    # Verschiebungsvektor von teilchen2 zu teilchen1
    verschiebung = teilchen1.position - teilchen2.position

    # Abstand zwischen Teilchen
    abstand = np.linalg.norm(verschiebung)

    # Für exakt überlappende Teilchen gib Nullkraft zurück
    if abstand < konst.EPSILON:
        return np.array([0.0, 0.0])

    # Soft-Core-Regularisierung für sehr nahe Teilchen
    # Verhindert numerische Explosion bei kleinen Abständen
    min_abstand = 1e-6
    if abstand < min_abstand:
        # Soft-Core-Potential: F = q1*q2*r/(r^2 + eps^2)^(3/2)
        eps = min_abstand
        r_quadrat = abstand * abstand
        nenner = (r_quadrat + eps * eps) ** 1.5
        kraft_betrag = (teilchen1.ladung * teilchen2.ladung * abstand) / nenner
        if abstand > 0:
            r_einheit = verschiebung / abstand
        else:
            return np.array([0.0, 0.0])
    else:
        # Normale Coulomb-Kraft für größere Abstände
        r_einheit = verschiebung / abstand
        kraft_betrag = (teilchen1.ladung * teilchen2.ladung) / (abstand ** 2)

    # Kraftvektor zeigt in Richtung der Verschiebung (abstoßend für gleiche Ladungen)
    kraft = kraft_betrag * r_einheit

    return kraft


def berechne_gesamte_elektrostatische_kraft(teilchen_index: int,
                                            teilchen: List[Teilchen]) -> np.ndarray:
    """
    Berechnet gesamte elektrostatische Kraft auf ein Teilchen von allen anderen.

    Summiert alle paarweisen Coulomb-Kräfte.

    Args:
        teilchen_index: Index des Teilchens
        teilchen: Liste aller Teilchen

    Returns:
        Gesamtkraftvektor
    """
    gesamt_kraft = np.array([0.0, 0.0])
    teilchen_i = teilchen[teilchen_index]

    # Summiere Kräfte von allen anderen Teilchen
    for j, teilchen_j in enumerate(teilchen):
        if j == teilchen_index:
            continue  # Keine Selbst-Interaktion

        paarweise_kraft = berechne_coulombkraft_zwischen(teilchen_i, teilchen_j)
        gesamt_kraft += paarweise_kraft

    return gesamt_kraft


def berechne_gesamtkraft(teilchen_index: int,
                         teilchen: List[Teilchen]) -> np.ndarray:
    """
    Berechnet Gesamtkraft auf ein Teilchen (Gravitation + Elektrostatisch).

    Args:
        teilchen_index: Index des Teilchens
        teilchen: Liste aller Teilchen

    Returns:
        Gesamtkraftvektor
    """
    aktuelles_teilchen = teilchen[teilchen_index]

    gravitationskraft = berechne_gravitationskraft(aktuelles_teilchen)
    elektrostatische_kraft = berechne_gesamte_elektrostatische_kraft(teilchen_index, teilchen)

    gesamtkraft = gravitationskraft + elektrostatische_kraft
    return gesamtkraft


def berechne_beschleunigung(teilchen_index: int,
                            teilchen: List[Teilchen]) -> np.ndarray:
    """
    Berechnet Beschleunigung eines Teilchens aus den Kräften.

    Verwendet Newtons zweites Gesetz: a = F/m

    Args:
        teilchen_index: Index des Teilchens
        teilchen: Liste aller Teilchen

    Returns:
        Beschleunigungsvektor [ax, ay]
    """
    aktuelles_teilchen = teilchen[teilchen_index]
    gesamtkraft = berechne_gesamtkraft(teilchen_index, teilchen)

    # Behandle Null-Masse Fall
    if abs(aktuelles_teilchen.masse) < konst.EPSILON:
        return np.array([0.0, 0.0])

    # Newtons zweites Gesetz: a = F/m
    beschleunigung = gesamtkraft / aktuelles_teilchen.masse
    return beschleunigung


def berechne_potentielle_energie_coulomb(teilchen: List[Teilchen]) -> float:
    """
    Berechnet gesamte Coulomb-Potentialenergie des Systems.

    U = (1/2) * sum_i sum_j (q_i * q_j / r_ij) für i != j

    Args:
        teilchen: Liste aller Teilchen

    Returns:
        Gesamte Coulomb-Potentialenergie
    """
    gesamt_potential = 0.0

    # Summiere über alle eindeutigen Paare (i < j um Doppelzählung zu vermeiden)
    for i in range(len(teilchen)):
        for j in range(i + 1, len(teilchen)):
            # Abstand zwischen Teilchen
            abstand = teilchen[i].abstand_zu(teilchen[j])

            # Gleiche Regularisierung wie bei Kraftberechnung verwenden
            min_abstand = 1e-6
            if abstand < min_abstand:
                # Soft-Core-Potential: U = q1*q2/sqrt(r^2 + eps^2)
                eps = min_abstand
                r_quadrat = abstand * abstand
                effektiver_abstand = np.sqrt(r_quadrat + eps * eps)
                paar_potential = (teilchen[i].ladung * teilchen[j].ladung) / effektiver_abstand
            else:
                # Normales Coulomb-Potential
                paar_potential = (teilchen[i].ladung * teilchen[j].ladung) / abstand

            gesamt_potential += paar_potential

    return gesamt_potential


def berechne_system_kraefte(teilchen: List[Teilchen]) -> List[np.ndarray]:
    """
    Berechnet Kräfte auf alle Teilchen im System.

    Args:
        teilchen: Liste aller Teilchen

    Returns:
        Liste der Kraftvektoren für jedes Teilchen
    """
    kraefte = []

    for i in range(len(teilchen)):
        kraft = berechne_gesamtkraft(i, teilchen)
        kraefte.append(kraft)

    return kraefte


def berechne_system_kraefte_symmetrisch(teilchen: List[Teilchen]) -> List[np.ndarray]:
    """
    Berechnet Kräfte auf alle Teilchen mit exakter Wahrung von Newtons 3. Gesetz.

    Stellt sicher dass Kräfte zwischen Teilchenpaaren exakt gleich und
    entgegengesetzt sind (actio = reactio).

    Args:
        teilchen: Liste aller Teilchen

    Returns:
        Liste der Kraftvektoren
    """
    n_teilchen = len(teilchen)
    kraefte = [np.array([0.0, 0.0]) for _ in range(n_teilchen)]

    # Gravitationskräfte hinzufügen
    for i in range(n_teilchen):
        kraefte[i] += berechne_gravitationskraft(teilchen[i])

    # Coulomb-Kräfte hinzufügen - jedes Paar nur einmal berechnen
    for i in range(n_teilchen):
        for j in range(i + 1, n_teilchen):
            # Berechne Kraft auf i durch j
            kraft_auf_i = berechne_coulombkraft_zwischen(teilchen[i], teilchen[j])

            # Wende gleiche und entgegengesetzte Kräfte an (3. Newtonsches Gesetz)
            kraefte[i] += kraft_auf_i
            kraefte[j] -= kraft_auf_i  # Gleich und entgegengesetzt

    return kraefte


def berechne_system_beschleunigungen(teilchen: List[Teilchen]) -> List[np.ndarray]:
    """
    Berechnet Beschleunigungen für alle Teilchen im System.

    Args:
        teilchen: Liste aller Teilchen

    Returns:
        Liste der Beschleunigungsvektoren
    """
    kraefte = berechne_system_kraefte_symmetrisch(teilchen)
    beschleunigungen = []

    for i, aktuelles_teilchen in enumerate(teilchen):
        if abs(aktuelles_teilchen.masse) < konst.EPSILON:
            beschleunigungen.append(np.array([0.0, 0.0]))
        else:
            # Newtons zweites Gesetz: a = F/m
            beschleunigungen.append(kraefte[i] / aktuelles_teilchen.masse)

    return beschleunigungen
