import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from utils.save_output import save_as_svg
from config import *

# Daten laden
train_list = load_file_list(TRAIN_FILE)
val_list = load_file_list(VAL_FILE)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = FloorPlanDataset(train_list, BASE_PATH, transform=train_transform)
val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Modell und Training
model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

# Beispiel: Speichern der Modellparameter
torch.save(model.state_dict(), "model_weights_2.pth")

# Validierung und SVG-Erstellung
model.eval()
with torch.no_grad():
    for imgs, _ in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        for i, pred in enumerate(preds):
            save_as_svg(pred, f"output/output_{i}.svg")
