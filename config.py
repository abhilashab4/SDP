import torch

# Project Configuration
PROJECT_NAME = "xerces" 
TRAIN_VERSIONS = ["1.2"]
TEST_VERSION = "1.3"

# Model Hyperparameters
METRICS_DIM = 21
VOCAB_SIZE = 5000
EMBED_DIM = 64
LATENT_DIM = 32

# Training Hyperparameters
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 200
KL_COEFF = 0.05

# Environment
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = f"models/{PROJECT_NAME}.pth"
VOCAB_SAVE_PATH = f"models/{PROJECT_NAME}_vocab.pkl"