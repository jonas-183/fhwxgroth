import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import DeepUNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from config import *

# Focal Loss für unbalancierte Daten
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, target):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(outputs, target)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss
        return focal_loss.mean()

# Ensemble-Modell (2 Modelle)
class EnsembleModel(nn.Module):
    def __init__(self, model1, model2):
        super(EnsembleModel, self).__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        softmax1 = torch.softmax(out1, dim=1)
        softmax2 = torch.softmax(out2, dim=1)
        return (softmax1 + softmax2) / 2  # Durchschnitt der Vorhersagen

# Lese Datei-Liste ein
val_list = load_file_list(VAL_FILE)

# Datenvorverarbeitung
val_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Original-Modell initialisieren
model_single = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)
model_single.load_state_dict(torch.load('model_weights_4.pth', map_location=DEVICE, weights_only = True))

# Ensemble-Modelle initialisieren
model1 = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)
model2 = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)

# Ensemble-Modell erstellen
ensemble_model = EnsembleModel(model1, model2).to(DEVICE)

# Modelle in den Evaluierungsmodus setzen
model_single.eval()
ensemble_model.eval()

# Focal Loss als Verlustfunktion
criterion = FocalLoss(gamma=2., alpha=0.25)

# Schleife über das Validierungs-Dataset
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Vorhersage mit dem Einzelmodell
        outputs_single = model_single(imgs)
        preds_single = torch.argmax(outputs_single, dim=1).cpu().numpy()

        # Vorhersage mit dem Ensemble-Modell
        outputs_ensemble = ensemble_model(imgs)
        preds_ensemble = torch.argmax(outputs_ensemble, dim=1).cpu().numpy()

        # Visualisierung der Eingabe und der Vorhersagen
        plt.figure(figsize=(20, 6))

        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(imgs[0].cpu().permute(1, 2, 0))
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("Original Prediction (Single Model)")
        plt.imshow(preds_single[0], cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("Ensemble Prediction")
        plt.imshow(preds_ensemble[0], cmap='gray')
        plt.axis('off')

        plt.tight_layout()
        plt.show()
        break
