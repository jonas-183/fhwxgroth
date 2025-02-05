import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.unet import DeepUNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from utils.save_output import save_as_svg
import matplotlib.pyplot as plt
from config import *

# Daten laden
train_list = load_file_list(TRAIN_FILE)
val_list = load_file_list(VAL_FILE)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),  # 50% Wahrscheinlichkeit für horizontales Spiegeln
    transforms.RandomRotation(degrees=10),  # Zufällige Rotation um bis zu 10°
    transforms.RandomAffine(degrees=0, shear=10),  # Zufälliges Scheren
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Helligkeit/Kontrast zufällig variieren
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisierung
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor()
])

train_dataset = FloorPlanDataset(train_list, BASE_PATH, transform=train_transform)
val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

def calculate_class_weights(dataset):
    class_counts = np.zeros(NUM_CLASSES)  # Anzahl der Pixel pro Klasse
    total_pixels = 0  # Gesamtanzahl der Pixel

    for _, labels in dataset:
        for class_id in range(NUM_CLASSES):
            class_counts[class_id] += (labels == class_id).sum().item()
        total_pixels += labels.numel()

    class_weights = total_pixels / (NUM_CLASSES * class_counts)
    return torch.tensor(class_weights, dtype=torch.float32)


class CombinedLoss(torch.nn.Module):
    def __init__(self, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.class_weights = class_weights

    def forward(self, outputs, targets):
        ce_loss = self.cross_entropy(outputs, targets)

        # Gewichteter Dice Loss
        smooth = 1.0
        outputs = torch.softmax(outputs, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (outputs * targets_one_hot).sum(dim=(2, 3))
        dice_loss = 1 - (2.0 * intersection + smooth) / (
                    outputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3)) + smooth)

        if self.class_weights is not None:
            dice_loss = dice_loss * self.class_weights.view(1, -1)  # Gewichte anwenden

        return ce_loss + dice_loss.mean()

# Modell und Training
#class_weights = calculate_class_weights(train_dataset)
#print(class_weights)
class_weights = torch.tensor([ 0.2836,  2.7955, 25.4555, 13.0407], dtype=torch.float32) # eingesetzte Werte kamen bei erster Berechnung mit calculate_class_weights heraus
class_weights = class_weights.to(DEVICE)
print(DEVICE)
model = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)
criterion = CombinedLoss(class_weights=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# Parameter für Early Stopping
patience = 5  # Anzahl der Epochen ohne Verbesserung
best_val_loss = float('inf')  # Beste Validierungsverlust initialisieren
counter = 0  # Zählt Epochen ohne Verbesserung

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Vorwärtsdurchlauf
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        # Rückwärtsdurchlauf
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)  # Durchschnittsverlust für die Epoche
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {train_loss:.4f}")

    # Validierung
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)  # Durchschnittsverlust für die Validierung
    print(f"Validation Loss: {val_loss:.4f}")

    # Early Stopping überprüfen
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "best_model_weights.pth")  # Bestes Modell speichern
        print("Validation loss improved. Saving model...")
    else:
        counter += 1
        print(f"No improvement in validation loss for {counter} epochs.")

        if counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    scheduler.step(val_loss)  # Lernrate anpassen basierend auf Validierungsverlust

# Modell nach Training speichern (falls nicht gestoppt)
if counter < patience:
    torch.save(model.state_dict(), "model_weights_1.pth")

# Pre Processing

#smoothed_preds =
# Validierung
model.eval()
# Anzeigen des Inputs und der Prediction
index_for_predcition = 0
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 4, 1)
        plt.title("Input")
        plt.imshow(imgs[index_for_predcition].cpu().permute(1, 2, 0))  # RGB-Darstellung
        plt.subplot(1, 4, 2)
        plt.title("Label")
        plt.imshow(labels[index_for_predcition].cpu(), cmap='gray')  # Ground Truth
        plt.subplot(1, 4, 3)
        plt.title("Prediction")
        plt.imshow(preds[index_for_predcition].cpu(), cmap='gray')  # Modellvorhersage
        plt.subplot(1, 4, 4)
        plt.title("Smoothed Prediction")
        plt.imshow(smoothed_preds[index_for_predcition].cpu(), cmap='gray')  # Modellvorhersage
        plt.show()
        break
