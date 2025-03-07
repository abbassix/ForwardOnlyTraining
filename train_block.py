"""
train_block.py

Trains a single specified block within a modular neural network.
Intermediate blocks are trained using forward-forward (positive vs. negative samples).
The final block is trained using standard supervised classification (CrossEntropyLoss).

Positive and negative augmentation is applied only for intermediate blocks.
"""

import logging
from dataclasses import dataclass

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
# Training Configuration Dataclass
# ------------------------------------------------------------------------------
@dataclass
class TrainConfig(DataConfig):
    data_gen_type: str = "random_noise"
    epochs: int = 5
    lr: float = 0.001
    optimizer: str = "adam"
    model: list = None
    num_classes: int = 10  # For MNIST/CIFAR-10; adjust accordingly

# ------------------------------------------------------------------------------
# Single block training function
# ------------------------------------------------------------------------------
def train_single_block(model, train_loader, cfg, block_index, device):
    model.to(device)
    block = model.blocks[block_index]
    block.train()

    is_final_block = (block_index == len(model.blocks) - 1)

    optimizer_cls = optim.Adam if cfg.optimizer.lower() == "adam" else optim.SGD
    optimizer = optimizer_cls(block.parameters(), lr=cfg.lr)

    # Loss functions for final vs intermediate blocks
    criterion = nn.CrossEntropyLoss() if is_final_block else nn.BCEWithLogitsLoss()

    for epoch in range(1, cfg.epochs + 1):
        epoch_loss, epoch_metric, batch_count = 0.0, 0.0, 0

        for batch in train_loader:
            images, labels = batch if isinstance(batch, (tuple, list)) else (batch, None)
            images = images.to(device)

            # Pass through preceding blocks without gradients
            with torch.no_grad():
                x = images
                for prev_block in model.blocks[:block_index]:
                    x = prev_block(x)

            optimizer_cls = optim.Adam if cfg.optimizer.lower() == "adam" else optim.SGD
            optimizer = optimizer_cls(block.parameters(), lr=cfg.lr)

            if is_final_block:
                # No augmentation, supervised training
                labels = labels.to(device)
                outputs = block(x)
                loss = criterion(outputs, labels)

                # Compute accuracy
                pred_labels = outputs.argmax(dim=1)
                accuracy = (pred_labels == labels).float().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({
                    f"final_block_batch_loss": loss.item(),
                    f"final_block_batch_accuracy": accuracy.item()
                })

                epoch_metric += accuracy.item()

            else:
                # Augmented data for forward-forward training
                augmented_x = generate_data(x, cfg, k=cfg.model[block_index][2])
                batch_size = x.size(0)

                outputs = block(augmented_x)

                outputs_flat = outputs.view(batch_size * 2, -1)
                goodness = torch.sum(outputs_flat ** 2, dim=-1)
                threshold = outputs_flat.shape[1]
                logits = goodness - threshold

                # Binary labels (1: positive, 0: negative)
                loss_labels = torch.zeros(batch_size * 2, device=device)
                loss_labels[:batch_size] = 1.0

                loss = criterion(logits, loss_labels)

                # Compute FF accuracy
                positive_goodness = goodness[:batch_size]
                ff_accuracy = (positive_goodness > threshold).float().mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({
                    f"block_{block_index+1}_batch_loss": loss.item(),
                    f"block_{block_index+1}_ff_accuracy": ff_accuracy.item()
                })

                epoch_metric += ff_accuracy.item()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_metric = epoch_metric / batch_count

        metric_name = "Accuracy" if is_final_block else "FF Accuracy"
        logger.info(
            f"Block [{block_index+1}], Epoch [{epoch}/{cfg.epochs}] - "
            f"Avg Loss: {avg_loss:.4f}, Avg {metric_name}: {avg_metric:.4f}"
        )

        wandb.log({
            f"block_{block_index+1}_epoch_avg_loss": avg_loss,
            f"block_{block_index+1}_epoch_avg_{metric_name.lower().replace(' ', '_')}": avg_metric
        })

# ------------------------------------------------------------------------------
# Main function (standalone test)
# ------------------------------------------------------------------------------
@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: TrainConfig):
    logger.info("Training single block with config:\n%s", OmegaConf.to_yaml(cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True), reinit=True)

    train_loader, _, _ = load_and_split_data(cfg)
    model = build_model(cfg)

    # Example: train first block; set explicitly for other blocks
    block_index = 0  # adjust as needed
    train_single_block(model, train_loader, cfg, block_index, device)

    wandb.finish()

if __name__ == "__main__":
    main()
