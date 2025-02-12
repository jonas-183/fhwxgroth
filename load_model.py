from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import UNet, DeepUNet, DeeperUNet
from utils.svg_utils import FloorPlanDataset, load_file_list
from utils.save_output import save_as_svg
import matplotlib.pyplot as plt
from config import *
import numpy as np
import torch
import cv2
from scipy.ndimage import gaussian_filter
from skimage.morphology import dilation, square, diamond, disk, closing

# Funktion zum Glätten der Vorhersage mit einem Gaußschen Filter
def smooth_prediction(prediction, sigma=1.0):
    return gaussian_filter(prediction, sigma=sigma)


# Funktion zum Vervollständigen der Vorhersage mit morphologischen Operationen
def fill_prediction(prediction, method="adaptive"):
    if method == "square":  # für stärkere füllung
        return dilation(prediction, square(4))
    elif method == "diamond":  # weichere glättung mit diamond
        return dilation(prediction, diamond(3))
    elif method == "disk":  # oder weichere glättung mit disk
        return dilation(prediction, disk(5))
    elif method == "closing":   # falls zu viel gefüllt wird, closing
        return closing(prediction, square(4))
    elif method == "adaptive":
        kernel_size = int(np.ceil(prediction.sum() / 1000))
        return dilation(prediction, square(max(3, kernel_size)))
    return prediction

val_list = load_file_list(VAL_FILE)

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = FloorPlanDataset(val_list, BASE_PATH, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Modellinitialisierung
model = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load('model_weights_4.pth', map_location=torch.device('cpu'), weights_only=True))

model.eval()

# Metrik-Initialisierung
correct_pixels = 0
total_pixels = 0
correct_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Klassenweise
total_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Pixelanzahl pro Klasse

# Schleife über das Validierungs-Dataset
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        softmaxed = torch.softmax(outputs, dim=1)
        confidence, preds = torch.max(softmaxed, dim=1)

        # Thresholding
        threshold = 0.5
        preds_thresholded = preds.clone()
        preds_thresholded[confidence < threshold] = 0

        # Konvertiere Vorhersagen
        preds_np = preds.cpu().numpy()
        preds_thresholded_np = preds_thresholded.cpu().numpy()

        # Glätten der Vorhersage
        smoothed_preds = np.zeros_like(preds_np, dtype=np.float32)
        for i in range(preds.size(0)):
            smoothed_preds[i] = smooth_prediction(preds_np[i], sigma=1.0)

        # Vervollständigung der Vorhersage
        filled_preds = np.zeros_like(smoothed_preds, dtype=np.float32)
        for i in range(smoothed_preds.shape[0]):
            filled_preds[i] = fill_prediction(smoothed_preds[i] > 0.6, method="square")  # davor adaptive

        # Pixelgenauigkeit berechnen
        correct_pixels += (preds == labels).sum().item()
        total_pixels += torch.numel(labels)

        # Klassenweise Genauigkeit berechnen
        for label, prediction in zip(labels.view(-1), preds.view(-1)):  # Pixelweise
            total_per_class[label.item()] += 1
            if label == prediction:
                correct_per_class[label.item()] += 1

        # **Adaptive Segmentierung**
        # Hier definieren wir eine Region anhand der Vorhersagen und Konfidenzwerte
        for i in range(imgs.size(0)):  # Batch-Verarbeitung
            img_pred = preds[i].cpu().numpy()  # Vorhersagebild (Tensor zu NumPy)
            img_conf = confidence[i].cpu().numpy()

            # Umwandlung in uint8, falls nötig
            img_pred = img_pred.astype(np.uint8)

            # Finde die Konturen in der Vorhersage
            contours, _ = cv2.findContours(img_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Erstelle eine Maske auf Basis der Konturen und Konfidenz
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Wenn die Konfidenz in diesem Bereich hoch genug ist, zeichne die Bounding Box
                if np.mean(img_conf[y:y + h, x:x + w]) > threshold:
                    cv2.rectangle(img_pred, (x, y), (x + w, y + h), (255), 1)  # Zeichne die Bounding Box

            # Visualisierung der adaptiven Segmentierung und der Prediction
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 6, 1)
            plt.title("Input")
            plt.imshow(imgs[i].cpu().permute(1, 2, 0))

            plt.subplot(1, 6, 2)
            plt.title("Label")
            plt.imshow(labels[i].cpu(), cmap='gray')

            plt.subplot(1, 6, 3)
            plt.title("Prediction")
            plt.imshow(preds[i].cpu(), cmap='gray')  # Modellvorhersage

            plt.subplot(1, 6, 6)
            plt.title("Adaptive Prediction")
            plt.imshow(img_pred, cmap='gray')

            plt.subplot(1, 6, 5)
            plt.title("Smoothed Prediction")
            plt.imshow(smoothed_preds[4], cmap='gray')

            plt.subplot(1, 6, 4)
            plt.title("Filled Prediction")
            plt.imshow(filled_preds[4], cmap='gray')

            '''block=False ist optional genau so wie plt.close()&plt.pause(1), 
            wenn nicht gewünscht dann auskommentieren und b=f rausnehmen'''
            plt.show(block=False)  # Nicht blockierend anzeigen, code läuft dadurch durch bis zum Ende
            plt.pause(1)  # 1 Sekunde warten
            plt.close()  # Fenster schließen

        break  # Nur die erste Batch visualisieren

# Gesamtaccuracy berechnen
overall_accuracy = 100 * correct_pixels / total_pixels
print(f'Overall Pixel Accuracy: {overall_accuracy:.2f} %')

# Klassenweise Genauigkeit berechnen
for classname, correct_count in correct_per_class.items():
    if total_per_class[classname] > 0:  # Vermeidung von Division durch 0
        class_accuracy = 100 * float(correct_count) / total_per_class[classname]
        print(f'Accuracy for class {classname} is {class_accuracy:.1f} %')
