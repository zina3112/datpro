"""
Plot- und Visualisierungsmodul

Bietet Visualisierungsfähigkeiten für die Simulation einschließlich
Energieplots, Teilchentrajektorien und Animationen.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
import os
from . import konstanten as konst
from .datenverwalter import Datenverwalter


class Visualisierer:
    """
    Visualisierungswerkzeuge für die Simulation geladener Teilchen.

    Bietet Methoden für:
    - Energieerhaltungsplots
    - Individuelle Teilchentrajektorien
    - Kombinierte Trajektorienplots
    - Phasenraumdiagramme
    """

    def __init__(self, datenverwalter: Datenverwalter, abbildungs_verzeichnis: Optional[str] = None):
        """
        Initialisiert den Visualisierer.

        Args:
            datenverwalter: Datenverwalter mit Simulationsdaten
            abbildungs_verzeichnis: Verzeichnis zum Speichern von Abbildungen
        """
        self.datenverwalter = datenverwalter

        # Setze Abbildungsverzeichnis
        if abbildungs_verzeichnis is None:
            abbildungs_verzeichnis = os.path.join(
                konst.AUSGABE_VERZEICHNIS, konst.PLOT_VERZEICHNIS)
        self.abbildungs_verzeichnis = abbildungs_verzeichnis

        # Erstelle Verzeichnis falls es nicht existiert
        os.makedirs(self.abbildungs_verzeichnis, exist_ok=True)

        # Setze Matplotlib-Stil
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Verwende Standard falls Stil nicht verfügbar
            pass

    def plotte_energie_vs_zeit(self, speichern: bool = True, anzeigen: bool = True) -> None:
        """
        Plottet Gesamtenergie versus Zeit zur Überprüfung der Erhaltung.

        Args:
            speichern: Ob Abbildung gespeichert werden soll
            anzeigen: Ob Abbildung angezeigt werden soll
        """
        zeiten, energien = self.datenverwalter.hole_energie_historie()

        if not zeiten:
            print("Keine Daten zum Plotten vorhanden")
            return

        # Berechne Energiedrift
        anfangsenergie = energien[0]
        relative_drift = [(e - anfangsenergie) / abs(anfangsenergie) * 100
                          for e in energien]

        # Erstelle Abbildung mit zwei Unterplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plotte absolute Energie
        ax1.plot(zeiten, energien, 'b-', linewidth=1.5, label='Gesamtenergie')
        ax1.axhline(y=anfangsenergie, color='r', linestyle='--',
                    alpha=0.5, label=f'Anfangsenergie = {anfangsenergie:.4f}')
        ax1.set_xlabel('Zeit (s)')
        ax1.set_ylabel('Gesamtenergie')
        ax1.set_title('Systemenergie in Abhängigkeit von der Zeit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plotte relative Energiedrift
        ax2.plot(zeiten, relative_drift, 'r-', linewidth=1.5)
        ax2.set_xlabel('Zeit (s)')
        ax2.set_ylabel('Energiedrift (%)')
        ax2.set_title('Relative Energiedrift')
        ax2.grid(True, alpha=0.3)

        # Füge finale Drift-Annotation hinzu
        finale_drift = relative_drift[-1]
        ax2.text(0.98, 0.98, f'Finale Drift: {finale_drift:.4f}%',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if speichern:
            dateiname = os.path.join(
                self.abbildungs_verzeichnis, 'energieerhaltung.png')
            plt.savefig(dateiname, dpi=konst.DPI, bbox_inches='tight')
            print(f"Energieplot gespeichert in {dateiname}")

        if anzeigen:
            plt.show()
        else:
            plt.close()

    def plotte_teilchen_trajektorie(self,
                                    teilchen_index: int,
                                    speichern: bool = True,
                                    anzeigen: bool = True) -> None:
        """
        Plottet Trajektorie eines einzelnen Teilchens.

        Args:
            teilchen_index: Index des zu plottenden Teilchens (0-basiert)
            speichern: Ob Abbildung gespeichert werden soll
            anzeigen: Ob Abbildung angezeigt werden soll
        """
        # Behandle ungültigen Index elegant
        try:
            x_positionen, y_positionen = self.datenverwalter.hole_teilchen_trajektorie(
                teilchen_index)
        except (ValueError, IndexError) as e:
            print(f"Kann Trajektorie nicht plotten: {e}")
            return

        if not x_positionen:
            print("Keine Trajektoriendaten zum Plotten vorhanden")
            return

        fig, ax = plt.subplots(figsize=konst.ABBILDUNGSGROESSE)

        # Plotte Trajektorie
        ax.plot(x_positionen, y_positionen, 'b-', linewidth=1, alpha=0.7)

        # Markiere Start- und Endpunkte
        ax.plot(x_positionen[0], y_positionen[0], 'go',
                markersize=10, label='Start', markeredgecolor='darkgreen')
        ax.plot(x_positionen[-1], y_positionen[-1], 'ro',
                markersize=10, label='Ende', markeredgecolor='darkred')

        # Zeichne Box-Grenzen
        self._zeichne_box_grenzen(ax)

        # Beschriftungen und Titel
        ax.set_xlabel('X-Position')
        ax.set_ylabel('Y-Position')
        ax.set_title(f'Bahn von Teilchen {teilchen_index + 1}')
        ax.legend()
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        if speichern:
            dateiname = os.path.join(self.abbildungs_verzeichnis,
                                     f'trajektorie_teilchen_{teilchen_index + 1}.png')
            plt.savefig(dateiname, dpi=konst.DPI, bbox_inches='tight')
            print(f"Trajektorienplot gespeichert in {dateiname}")

        if anzeigen:
            plt.show()
        else:
            plt.close()

    def plotte_alle_trajektorien(self, speichern: bool = True, anzeigen: bool = True) -> None:
        """
        Plottet Trajektorien aller Teilchen in derselben Abbildung.

        Args:
            speichern: Ob Abbildung gespeichert werden soll
            anzeigen: Ob Abbildung angezeigt werden soll
        """
        trajektorien = self.datenverwalter.hole_alle_trajektorien()

        if not trajektorien:
            print("Keine Trajektoriendaten zum Plotten vorhanden")
            return

        fig, ax = plt.subplots(figsize=(12, 12))

        # Farbkarte für verschiedene Teilchen
        farben = plt.cm.rainbow(np.linspace(0, 1, len(trajektorien)))

        # Plotte jede Teilchentrajektorie
        for i, (x_pos, y_pos) in enumerate(trajektorien):
            if not x_pos:  # Überspringe leere Trajektorien
                continue

            ax.plot(x_pos, y_pos, color=farben[i], linewidth=1,
                    alpha=0.6, label=f'Teilchen {i + 1}')

            # Markiere Startposition
            ax.plot(x_pos[0], y_pos[0], 'o', color=farben[i],
                    markersize=8, markeredgecolor='black')

            # Markiere Endposition
            ax.plot(x_pos[-1], y_pos[-1], 's', color=farben[i],
                    markersize=8, markeredgecolor='black')

        # Zeichne Box-Grenzen
        self._zeichne_box_grenzen(ax)

        # Beschriftungen und Titel
        ax.set_xlabel('X-Position')
        ax.set_ylabel('Y-Position')
        ax.set_title('Bahnen aller Teilchen')
        if len(trajektorien) <= 10:  # Zeige Legende nur für vernünftige Anzahl
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if speichern:
            dateiname = os.path.join(
                self.abbildungs_verzeichnis, 'alle_trajektorien.png')
            plt.savefig(dateiname, dpi=konst.DPI, bbox_inches='tight')
            print(f"Kombinierter Trajektorienplot gespeichert in {dateiname}")

        if anzeigen:
            plt.show()
        else:
            plt.close()

    def _zeichne_box_grenzen(self, ax) -> None:
        """
        Zeichnet Box-Grenzen auf dem Plot.

        Args:
            ax: Matplotlib-Achsenobjekt
        """
        # Zeichne Box-Wände
        box_x = [konst.BOX_MIN_X, konst.BOX_MAX_X, konst.BOX_MAX_X,
                 konst.BOX_MIN_X, konst.BOX_MIN_X]
        box_y = [konst.BOX_MIN_Y, konst.BOX_MIN_Y, konst.BOX_MAX_Y,
                 konst.BOX_MAX_Y, konst.BOX_MIN_Y]

        ax.plot(box_x, box_y, 'k-', linewidth=2, label='Box-Grenze')

        # Setze Achsenlimits mit kleinem Rand
        rand = 5
        ax.set_xlim(konst.BOX_MIN_X - rand, konst.BOX_MAX_X + rand)
        ax.set_ylim(konst.BOX_MIN_Y - rand, konst.BOX_MAX_Y + rand)

    def erstelle_zusammenfassungsbericht(self) -> None:
        """Erstellt umfassenden Zusammenfassungsbericht mit allen Plots."""
        print("\nGeneriere Zusammenfassungsbericht...")

        # Energieerhaltungsplot
        self.plotte_energie_vs_zeit(speichern=True, anzeigen=False)

        # Individuelle Teilchentrajektorien
        for i in range(konst.N_TEILCHEN):
            self.plotte_teilchen_trajektorie(i, speichern=True, anzeigen=False)

        # Kombinierte Trajektorien
        self.plotte_alle_trajektorien(speichern=True, anzeigen=False)

        # Statistiken
        statistiken = self.datenverwalter.hole_statistiken()

        # Schreibe Statistiken in Datei
        statistik_datei = os.path.join(
            self.abbildungs_verzeichnis, 'statistiken.txt')
        with open(statistik_datei, 'w') as f:
            f.write("SIMULATIONSSTATISTIKEN\n")
            f.write("=" * 50 + "\n\n")

            for schluessel, wert in statistiken.items():
                f.write(f"{schluessel}: {wert}\n")

        print(f"Zusammenfassungsbericht gespeichert in {
              self.abbildungs_verzeichnis}")
