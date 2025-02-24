from torch.utils.data import DataLoader
from torchvision import transforms
from models.unet import DeepUNet
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

model.load_state_dict(torch.load('best_model_weights_3.pth', map_location=torch.device('cpu'), weights_only=True))

model.eval()

# Metrik-Initialisierung
correct_pixels = 0
total_pixels = 0
correct_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Klassenweise
total_per_class = {classname: 0 for classname in range(NUM_CLASSES)}  # Pixelanzahl pro Klasse

# Collect all predictions and images
all_imgs = []
all_labels = []
all_preds = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        # Store batch data
        all_imgs.extend(imgs.cpu())
        all_labels.extend(labels.cpu())
        all_preds.extend(preds.cpu())

# Create interactive plot
current_idx = [0]  # Using list to allow modification in button callback

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
plt.subplots_adjust(bottom=0.2)  # Make room for buttons


def update_plot(idx):
    for ax in axs:
        ax.clear()

    axs[0].imshow(all_imgs[idx].permute(1, 2, 0))
    axs[0].set_title("Input")
    axs[1].imshow(all_labels[idx], cmap='gray')
    axs[1].set_title("Label")
    axs[2].imshow(all_preds[idx], cmap='gray')
    axs[2].set_title("Prediction")
    plt.draw()


def on_prev_button(event):
    current_idx[0] = (current_idx[0] - 1) % len(all_imgs)
    update_plot(current_idx[0])


def on_next_button(event):
    current_idx[0] = (current_idx[0] + 1) % len(all_imgs)
    update_plot(current_idx[0])


# Add buttons
ax_prev = plt.axes([0.3, 0.05, 0.1, 0.075])
ax_next = plt.axes([0.6, 0.05, 0.1, 0.075])
btn_prev = plt.Button(ax_prev, 'Previous')
btn_next = plt.Button(ax_next, 'Next')

btn_prev.on_clicked(on_prev_button)
btn_next.on_clicked(on_next_button)

# Show initial plot
update_plot(0)
plt.show()

