"""
train_block.py

This module loads a model and data, generates augmented data by concatenating
the original batch (positive samples) with random noise (negative samples),
computes a binary classification loss using BCEWithLogitsLoss, and updates the
model parameters via an optimizer (SGD or Adam).

The augmented batch is created as follows:
    - positive samples: original data (first half)
    - negative samples: random noise (second half)

The logits are computed by flattening the model outputs, calculating the "goodness"
as the sum of squared activations, and subtracting a threshold (the flattened dimension).
Binary labels are 1 for positive samples and 0 for negative samples.
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
from omegaconf import OmegaConf

# Import data and model functions
from data import load_and_split_data, Config as DataConfig
from data_gen import generate_data
from blocks import PoolConv

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration Dataclass for Training
# ------------------------------------------------------------------------------
@dataclass
class TrainConfig(DataConfig):
    # Data generation type (e.g., "random_noise")
    data_gen_type: str = "random_noise"
    # Training parameters
    epochs: int = 5
    lr: float = 0.001
    optimizer: str = "adam"  # Options: "adam", "sgd"
    # Model parameters for PoolConv
    c_in: int = 1     # For MNIST use 1, for CIFAR use 3; will adjust based on dataset.
    m: int = 2        # Pooling kernel size (if m == 1, no pooling is applied)
    c_out: int = 16   # Number of output channels
    k: int = 3        # Convolution kernel size

# ------------------------------------------------------------------------------
# Training Function
# ------------------------------------------------------------------------------
@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg: TrainConfig) -> None:
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Initialize wandb logging
    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True), reinit=True)
    
    # Adjust input channels based on dataset type
    if cfg.dataset.lower() == "mnist":
        model_c_in = 1
    else:
        model_c_in = cfg.c_in  # Expect 3 for CIFAR10/CIFAR100
    
    # Initialize model, loss function, and optimizer
    model = PoolConv(c_in=model_c_in, m=cfg.m, c_out=cfg.c_out, k=cfg.k).to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    if cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    
    # Load data (training, validation, test)
    train_loader, _, _ = load_and_split_data(cfg)
    
    # Training loop
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Extract images (assume batch is either tensor or tuple (images, labels))
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            
            batch_size = images.size(0)
            images = images.to(device)
            
            # Generate augmented data (concatenates original with random noise)
            # New batch shape: (2 * batch_size, C, H, W)
            new_images = generate_data(images, cfg).to(device)
            
            # Forward pass through the model
            outputs = model(new_images)
            outputs_flat = outputs.view(batch_size * 2, -1)
            goodness = torch.sum(outputs_flat ** 2, dim=-1)
            threshold = outputs_flat.shape[1]
            logits = goodness - threshold
            
            # Compute FF accuracy for positive samples
            positive_goodness = goodness[:batch_size]
            ff_accuracy = (positive_goodness > threshold).float().mean()
            
            # Create binary labels: 1 for positive samples, 0 for negative samples
            loss_labels = torch.zeros(batch_size * 2, device=device)
            loss_labels[:batch_size] = 1.0
            
            # Compute loss and update parameters
            loss = criterion(logits, loss_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_accuracy += ff_accuracy.item()
            batch_count += 1
            
            wandb.log({"batch_loss": loss.item()})
            wandb.log({"batch_ff_accuracy": ff_accuracy.item()})
        
        avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        avg_accuracy = epoch_accuracy / batch_count if batch_count > 0 else 0.0
        logger.info("Epoch [%d/%d] - Average Loss: %.4f, Average FF Accuracy: %.4f", epoch, cfg.epochs, avg_loss, avg_accuracy)
        wandb.log({"epoch": epoch, "avg_loss": avg_loss, "avg_accuracy": avg_accuracy})
    
    wandb.finish()

if __name__ == "__main__":
    main()
