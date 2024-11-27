import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from utils.svg_utils import FloorPlanDataset, load_file_list  # Importiere deine Dataset-Klasse und Hilfsfunktionen

def visualize(img, label):
    """
    Visualisiert das Bild und das zugehörige Label nebeneinander.
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Bild anzeigen
    ax[0].imshow(np.transpose(img.numpy(), (1, 2, 0)))  # Umwandeln von Tensor zu NumPy (HWC)
    ax[0].set_title("Bild")
    ax[0].axis('off')

    # Label anzeigen (wir gehen davon aus, dass das Label eine Maske ist)
    ax[1].imshow(label.numpy(), cmap='tab20b')  # Benutze einen Farb-Map, der gut zu den Labels passt
    ax[1].set_title("Label")
    ax[1].axis('off')

    plt.show()

def test_dataset():
    # Beispiel-Pfade
    train_file = "data/train.txt"
    base_path = "data/cubicasa5k"

    # Lade die Datei-Liste
    file_list = load_file_list(train_file)

    # Transformationen für das Bild
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    # Dataset erstellen
    dataset = FloorPlanDataset(file_list, base_path, transform)

    # DataLoader für das Dataset
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Testen: Lade ein Batch aus dem DataLoader
    for idx, (img, label) in enumerate(data_loader):
        print(f"Batch {idx + 1}:")
        print("Bildgröße:", img.shape)
        print("Labelgröße:", label.shape)
        print("Einzigartige Labelwerte:", torch.unique(label))

        # Visualisieren
        visualize(img[0], label[0])  # Visualisiere das erste Bild und das zugehörige Label

        if idx >= 2:  # Beispiel: Stoppe nach den ersten 3 Batches
            break
test_dataset()

