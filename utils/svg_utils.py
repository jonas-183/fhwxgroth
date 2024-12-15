import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from xml.dom import minidom
from skimage.draw import polygon


class FloorPlanDataset(Dataset):
    def __init__(self, file_list, base_path, transform=None):
        """
        Dataset zur Verarbeitung von Grundrissbildern und zugehörigen SVG-Labels.
        :param file_list: Liste von Ordnernamen (aus train.txt oder val.txt).
        :param base_path: Basisverzeichnis für die Daten (z. B. 'data/cubicasa5k').
        :param transform: Transformationen für das Bild.
        """
        self.file_list = file_list
        self.base_path = base_path
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        folder = self.file_list[idx].strip()
        # Konstruiere Pfade relativ zu `base_path`
        img_path = self.base_path + folder + 'F1_original.png'
        label_path = self.base_path + folder + 'model.svg'

        # Lade das Eingangsbild
        img = Image.open(img_path).convert('RGB')

        width, height = img.size

        # Lade und verarbeite die Labels aus der SVG-Datei
        label = self._load_label(label_path, width, height)

        # Wende Transformationen an (falls vorhanden)
        if self.transform:
            img = self.transform(img)

        # Konvertiere das Label explizit zu einem Tensor
        label = torch.tensor(label, dtype=torch.long)

        return img, label

    def _load_label(self, label_path, width, height):
        """
        Konvertiere eine SVG-Datei in ein 2D-Array, das Segmentierungslabels enthält.
        :param label_path: Pfad zur SVG-Datei.
        :return: Numpy-Array mit Labels (0: Hintergrund, 1: Wand, 2: Tür, 3: Fenster).
        """
        # Initialisiere ein leeres Array für die Labels (256x256 für konsistente Größe)
        label = np.zeros((256, 256), dtype=np.uint8)

        # Parsen der SVG-Datei
        svg = minidom.parse(label_path)

        svg_tag = svg.getElementsByTagName('svg')[0]
        viewBox = svg_tag.getAttribute('viewBox')
        if not viewBox:
            raise ValueError("Keine viewBox in der SVG-Datei gefunden.")

        _, _, viewBox_width, viewBox_height = map(float, viewBox.split())

        # Gehe durch die SVG-Elemente und ordne Farben zu Klassen
        for e in svg.getElementsByTagName('g'):  # Alle Gruppen durchsuchen
            # Erkenne Wände
            if e.getAttribute("id") == "Wall":
                self._process_wall(e, label, viewBox_width, viewBox_height)

            # Erkenne Fenster
            elif e.getAttribute("id") == "Window":
                self._process_window(e, label, viewBox_width, viewBox_height)

            # Erkenne Türen
            elif e.getAttribute("id") == "Door":
                self._process_door(e, label, viewBox_width, viewBox_height)

            continue

        return torch.tensor(label, dtype=torch.long)

    def _process_wall(self, e, label, viewBox_width, viewBox_height):
        """
        Verarbeite Wand-Elemente in der SVG-Datei und setze die Label-Region.
        :param e: Das XML-Element für die Wand.
        :param label: Das Label-Array.
        """
        # Hier kannst du den Code zur Wand-Erkennung anpassen
        # Beispiel: Wand als schwarze Polygone
        X, Y = get_points(e, viewBox_width, viewBox_height)  # Hole die Koordinaten
        rr, cc = polygon(Y,X)  # Berechne Pixelkoordinaten
        rr = np.clip(rr, 0, label.shape[0] - 1)
        cc = np.clip(cc, 0, label.shape[1] - 1)
        label[rr, cc] = 1  # Setze Wand-Label (1)

    def _process_window(self, e, label, viewBox_width, viewBox_height):
        """
        Verarbeite Fenster-Elemente in der SVG-Datei und setze die Label-Region.
        :param e: Das XML-Element für das Fenster.
        :param label: Das Label-Array.
        """
        # Beispiel: Fenster als grüne Polygone
        X, Y = get_points(e, viewBox_width, viewBox_height)  # Hole die Koordinaten
        rr, cc = polygon(Y,X)  # Berechne Pixelkoordinaten
        rr = np.clip(rr, 0, label.shape[0] - 1)
        cc = np.clip(cc, 0, label.shape[1] - 1)
        label[rr, cc] = 3  # Setze Fenster-Label (3)

    def _process_door(self, e, label, viewBox_width, viewBox_height):
        """
        Verarbeite Tür-Elemente in der SVG-Datei und setze die Label-Region.
        :param e: Das XML-Element für die Tür.
        :param label: Das Label-Array.
        """
        # Beispiel: Türen als blaue Polygone
        X, Y = get_points(e, viewBox_width, viewBox_height)  # Hole die Koordinaten
        rr, cc = polygon(Y,X)  # Berechne Pixelkoordinaten
        rr = np.clip(rr, 0, label.shape[0] - 1)
        cc = np.clip(cc, 0, label.shape[1] - 1)
        label[rr, cc] = 2  # Setze Tür-Label (2)


def load_file_list(file_path):
    """
    Lade eine Liste von Ordnernamen aus einer Datei (z. B. train.txt oder val.txt).
    :param file_path: Pfad zur Datei.
    :return: Liste von Ordnernamen.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines


# Hilfsfunktionen
def get_points(e, viewBox_width, viewBox_height):
    """
    Hilfsfunktion zum Extrahieren der Koordinaten von SVG-Elementen.
    :param e: Das SVG-Element.
    :return: Listen von X- und Y-Koordinaten.
    """
    # Implementiere hier die Logik zum Extrahieren der X- und Y-Koordinaten der SVG-Elemente
    # Beispiel: Gehe durch die "points"-Attribut, falls vorhanden
    pol = next(p for p in e.childNodes if p.nodeName == "polygon")
    points = pol.getAttribute("points").split(' ')
    points = points[:-1]

    X, Y = np.array([]), np.array([])
    for a in points:
        x, y = map(float, a.split(','))
        X = np.append(X, x)
        Y = np.append(Y, y)

    # SVG-Koordinaten auf die Größe des Labels skalieren
    X = np.round(X * (256 / viewBox_width)).astype(int)  # 1024 ist die SVG-Basisgröße
    Y = np.round(Y * (256 / viewBox_height)).astype(int)

    return X, Y
