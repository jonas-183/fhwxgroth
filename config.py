# Konfigurationen für das Projekt
import torch

TRAIN_FILE = "C:/Users/r1jon/PycharmProjects/fhwxgroth/data/train.txt"
VAL_FILE = "C:/Users/r1jon/PycharmProjects/fhwxgroth/data/val.txt"
BASE_PATH = "C:/Users/r1jon/PycharmProjects/fhwxgroth/data/cubicasa5k"
BATCH_SIZE = 16
NUM_CLASSES = 4
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

