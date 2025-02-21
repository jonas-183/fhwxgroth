from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import DeepUNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from utils.save_output import save_as_svg
import matplotlib.pyplot as plt
from config import *
import numpy as np
import torch
import cv2

from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, closing, square, remove_small_objects


# Verbesserte Funktion zur adaptiven Vorhersage-Optimierung
def adaptive_prediction(prediction, min_size=50):
    # Canny-Edge-Detection zur Strukturerhaltung
    edges = cv2.Canny((prediction * 255).astype(np.uint8), 50, 150)
    edges_dilated = dilation(edges, square(2))

    # Entferne sehr kleine isolierte Bereiche
    filtered = remove_small_objects(prediction.astype(bool), min_size=min_size)

    # Morphologische Schließung, um Blockstrukturen zu bewahren
    processed = closing(filtered, square(3))

    return np.maximum(processed, edges_dilated)


# Lade Validierungsdaten
val_list = load_file_list(VAL_FILE)

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modell laden
model = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('best_model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Evaluation
correct_pixels, total_pixels = 0, 0
correct_per_class = {classname: 0 for classname in range(NUM_CLASSES)}
total_per_class = {classname: 0 for classname in range(NUM_CLASSES)}

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        outputs = model(imgs)
        softmaxed = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(softmaxed, dim=1)

        # Threshold setzen
        threshold = 0.5
        preds[confidence < threshold] = 0

        preds_np = preds.cpu().numpy()

        # Optimierte adaptive Vorhersage
        refined_preds = np.zeros_like(preds_np, dtype=np.float32)

        for i in range(preds.size(0)):
            refined_preds[i] = adaptive_prediction(preds_np[i] > 0.6)

        # Berechnung der Gesamtgenauigkeit
        correct_pixels += (preds == labels).sum().item()
        total_pixels += torch.numel(labels)

        for label, prediction in zip(labels.view(-1), preds.view(-1)):
            total_per_class[label.item()] += 1
            if label == prediction:
                correct_per_class[label.item()] += 1

        # Visualisierung mit Labelling
        for i in range(imgs.size(0)):
            img_pred = preds[i].cpu().numpy().astype(np.uint8)
            img_conf = confidence[i].cpu().numpy()

            # Konturen finden für Labelling
            contours, _ = cv2.findContours(img_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_labeled = cv2.cvtColor(img_pred * 255, cv2.COLOR_GRAY2BGR)

            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                class_label = int(torch.mode(preds[i, y:y + h, x:x + w].flatten())[0].item())

                cv2.rectangle(img_labeled, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img_labeled, f'Class {class_label}', (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            plt.figure(figsize=(12, 5))
            plt.subplot(1, 5, 1)
            plt.title("Input")
            plt.imshow(imgs[i].cpu().permute(1, 2, 0))

            plt.subplot(1, 5, 2)
            plt.title("Prediction")
            plt.imshow(preds[i].cpu(), cmap='gray')

            plt.subplot(1, 5, 3)
            plt.title("Refined Prediction")
            plt.imshow(refined_preds[i], cmap='gray')

            plt.subplot(1, 5, 4)
            plt.title("Ground Truth")
            plt.imshow(labels[i].cpu(), cmap='gray')

            plt.subplot(1, 5, 5)
            plt.title("Labeled Prediction")
            plt.imshow(img_labeled)

            plt.show()
            break  # Beende nach einer Visualisierung

# Gesamtgenauigkeit berechnen
overall_accuracy = 100 * correct_pixels / total_pixels
print(f'Overall Pixel Accuracy: {overall_accuracy:.2f} %')

# Klassenweise Genauigkeit
for classname, correct_count in correct_per_class.items():
    if total_per_class[classname] > 0:
        class_accuracy = 100 * float(correct_count) / total_per_class[classname]
        print(f'Accuracy for class {classname}: {class_accuracy:.1f} %')
