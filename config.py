# Konfigurationen für das Projekt
import torch

TRAIN_FILE = "data/train.txt"
VAL_FILE = "data/val.txt"
BASE_PATH = "data/cubicasa5k"
BATCH_SIZE = 32
NUM_CLASSES = 4
EPOCHS = 50
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

