from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet, DeepUNet, DeeperUNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from utils.save_output import save_as_svg
import matplotlib.pyplot as plt
from config import *

val_list = load_file_list(VAL_FILE)

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=True)

# Modellinitialisierung
model = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load('model_weights_4.pth', map_location=torch.device('cpu'), weights_only=True))

model.eval()

# Metrik-Initialisierung
correct_pixels = 0
total_pixels = 0
correct_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Klassenweise
total_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Pixelanzahl pro Klasse

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Modellvorhersagen
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        # Pixelgenauigkeit berechnen
        correct_pixels += (preds == labels).sum().item()
        total_pixels += torch.numel(labels)

        # Klassenweise Genauigkeit berechnen
        for label, prediction in zip(labels.view(-1), preds.view(-1)):  # Pixelweise
            total_per_class[label.item()] += 1
            if label == prediction:
                correct_per_class[label.item()] += 1

        # Beispielvisualisierung (optional)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Input")
        plt.imshow(imgs[4].cpu().permute(1, 2, 0))  # RGB-Darstellung
        plt.subplot(1, 3, 2)
        plt.title("Label")
        plt.imshow(labels[4].cpu(), cmap='gray')  # Ground Truth
        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(preds[4].cpu(), cmap='gray')  # Modellvorhersage
        plt.show()
        break  # Nur die erste Batch visualisieren

# Gesamtaccuracy berechnen
overall_accuracy = 100 * correct_pixels / total_pixels
print(f'Overall Pixel Accuracy: {overall_accuracy:.2f} %')

# Klassenweise Genauigkeit berechnen
for classname, correct_count in correct_per_class.items():
    if total_per_class[classname] > 0:  # Vermeidung von Division durch 0
        class_accuracy = 100 * float(correct_count) / total_per_class[classname]
        print(f'Accuracy for class {classname} is {class_accuracy:.1f} %')
