# config.py
import torch
from pathlib import Path

# ---- Data Generation Parameters ----
# Create *known* parameters
WEIGHT = 0.7
BIAS = 0.3

# Create
START = 0
END = 1
STEP = 0.02

# ---- Train/Test Split ----
TRAIN_SPLIT_RATIO = 0.8  # Changed from train_split to a ratio for flexibility

# ---- Model & Training Parameters ----
RANDOM_SEED = 42
EPOCHS = 200
LEARNING_RATE = 0.01

# ---- Model Saving ----
MODEL_PATH = Path("models")
MODEL_NAME = "01_pytorch_workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# ---- Device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
