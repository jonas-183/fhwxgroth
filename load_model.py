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

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

#model = UNet(num_classes=NUM_CLASSES).to(DEVICE)
model = DeepUNet(num_classes=NUM_CLASSES).to(DEVICE)
#model = DeeperUNet(num_classes=NUM_CLASSES).to(DEVICE)

model.load_state_dict(torch.load('model_weights_4.pth'))

model.eval()
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

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
        break