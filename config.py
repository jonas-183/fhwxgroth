# Konfigurationen für das Projekt
import torch

TRAIN_FILE = "data/train.txt"
VAL_FILE = "data/val.txt"
BASE_PATH = "data/cubicasa5k"
BATCH_SIZE = 8
NUM_CLASSES = 4
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

