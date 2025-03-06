"""
train_block.py

Contains a reusable function `train_single_block` that trains a specified block of a modular neural network.
It receives the model, data loader, configuration, block index, and computation device as parameters.

This module can be run independently for testing or imported and used from other training scripts.
"""

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import hydra
from omegaconf import OmegaConf

from data import load_and_split_data, Config as DataConfig
from data_gen import generate_data
from model_factory import build_model

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Configuration Dataclass
# ------------------------------------------------------------------------------
from dataclasses import dataclass

@dataclass
class TrainConfig(DataConfig):
    data_gen_type: str = "random_noise"
    epochs: int = 5
    lr: float = 0.001
    optimizer: str = "adam"
    model: list = None

# ------------------------------------------------------------------------------
# Training function for a single block
# ------------------------------------------------------------------------------
def train_single_block(model, train_loader, cfg, block_index, device):
    """
    Trains a single block (specified by block_index) of the given modular model.

    Args:
        model (nn.Module): Modular model with blocks.
        train_loader (DataLoader): Data loader for training data.
        cfg (TrainConfig): Configuration containing training parameters.
        block_index (int): Index of the block to train.
        device (torch.device): Computation device.
    """
    model.to(device)
    block = model.blocks[block_index]
    block.train()

    # Choose optimizer
    optimizer_cls = optim.Adam if cfg.optimizer.lower() == "adam" else optim.SGD
    optimizer = optimizer_cls(block.parameters(), lr=cfg.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        batch_count = 0

        for batch in train_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            images = images.to(device)
            batch_size = images.size(0)

            # Forward pass through previous blocks without gradient tracking
            with torch.no_grad():
                x = images
                for prev_block in model.blocks[:block_index]:
                    x = prev_block(x)
                    
            # Kernel size for patch shuffle data generation
            k = cfg.model[block_index][2] if cfg.data_gen_type == "patch_shuffle" else None

            # Augment data immediately before the current block
            augmented_x = generate_data(x, cfg, k=k)

            # Forward pass through current block
            outputs = block(augmented_x)

            # Compute logits
            outputs_flat = outputs.view(batch_size * 2, -1)
            goodness = torch.sum(outputs_flat ** 2, dim=-1)
            threshold = outputs_flat.shape[1]
            logits = goodness - threshold

            # Binary labels: 1 for positive samples, 0 for negative samples
            loss_labels = torch.zeros(batch_size * 2, device=device)
            loss_labels[:batch_size] = 1.0

            # Compute loss and accuracy
            loss = criterion(logits, loss_labels)
            positive_goodness = goodness[:batch_size]
            ff_accuracy = (positive_goodness > threshold).float().mean()

            # Backpropagation and parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_accuracy += ff_accuracy.item()
            batch_count += 1

            # Log batch metrics
            wandb.log({
                f"block_{block_index+1}_batch_loss": loss.item(),
                f"block_{block_index+1}_batch_ff_accuracy": ff_accuracy.item()
            })

        avg_loss = epoch_loss / batch_count
        avg_accuracy = epoch_accuracy / batch_count

        logger.info(
            "Block [%d], Epoch [%d/%d] - Avg Loss: %.4f, Avg FF Accuracy: %.4f",
            block_index + 1, epoch, cfg.epochs, avg_loss, avg_accuracy
        )

        # Log epoch metrics
        wandb.log({
            f"block_{block_index+1}_epoch_avg_loss": avg_loss,
            f"block_{block_index+1}_epoch_avg_ff_accuracy": avg_accuracy
        })

# ------------------------------------------------------------------------------
# Main function (for standalone testing)
# ------------------------------------------------------------------------------
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    logger.info("Starting standalone block training:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Initialize wandb
    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True), reinit=True)

    train_loader, _, _ = load_and_split_data(cfg)
    model = build_model(cfg)

    # For standalone testing, train only the first block (index 0)
    train_single_block(model, train_loader, cfg, block_index=0, device=device)

    wandb.finish()

if __name__ == "__main__":
    main()
