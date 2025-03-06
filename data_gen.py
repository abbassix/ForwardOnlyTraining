"""
data_gen.py

This module loads the data using the load_and_split_data function from data.py,
and then generates augmented data according to the configuration.
The default data generation method, "random_noise", takes a batch of data with B samples
and concatenates it with B samples of random white noise (with shape (B, C, W, H)),
resulting in a new batch with shape (2*B, C, W, H).
"""

import logging
import torch
import wandb
from omegaconf import OmegaConf
from dataclasses import dataclass
import hydra
from typing import Optional

# Import the data loading function and base configuration from data.py
from data import load_and_split_data, Config as DataConfig

logger = logging.getLogger(__name__)

@dataclass
class DataGenConfig(DataConfig):
    # Specify the data generation type. Default is "random_noise".
    data_gen_type: str = "random_noise"


def generate_patch_shuffled_negatives(batch: torch.Tensor, k: int) -> torch.Tensor:
    """
    Generate negative samples by minimally perturbing images. 
    Within each group of (2k-2) rows and columns, swap the first (k-1) 
    rows and columns with the next (k-1).

    Args:
        batch (Tensor): shape (B, C, H, W), input batch of images.
        k (int): Kernel size parameter determining block size.

    Returns:
        Tensor: Negative samples tensor with shape (B, C, H, W).
    """
    B, C, H, W = batch.shape
    block_size = 2 * k - 2
    half_block = k - 1

    # Initialize index arrays for rows and columns
    row_indices = torch.arange(H)
    col_indices = torch.arange(W)

    # Reorder rows within each block
    for start_row in range(0, H - half_block, block_size):
        end_row = min(start_row + block_size, H)
        mid_row = start_row + half_block
        if mid_row < end_row:
            # Swap segments [start:mid] and [mid:end]
            row_indices[start_row:end_row] = torch.cat((
                row_indices[mid_row:end_row],
                row_indices[start_row:mid_row]
            ))

    # Reorder columns within each block
    for start_col in range(0, W - half_block, block_size):
        end_col = min(start_col + block_size, W)
        mid_col = start_col + half_block
        if mid_col < end_col:
            # Swap segments [start:mid] and [mid:end]
            col_indices[start_col:end_col] = torch.cat((
                col_indices[mid_col:end_col],
                col_indices[start_col:mid_col]
            ))

    # Apply the reordered indices to batch directly
    negatives = batch[:, :, row_indices, :]
    negatives = negatives[:, :, :, col_indices]

    return negatives


def generate_data(batch: torch.Tensor, cfg: DataGenConfig, k: Optional[int] = None) -> torch.Tensor:
    """
    Generate new data based on the specified method in the config.
    
    For "random_noise": Given a batch of shape (B, C, W, H),
    generate B samples of white noise with the same shape and concatenate them,
    resulting in an output batch of shape (2*B, C, W, H).
    """
    if cfg.data_gen_type == "random_noise":
        negatives = torch.randn_like(batch)
    elif cfg.data_gen_type == "patch_shuffle":
        negatives = generate_patch_shuffled_negatives(batch, k)
    else:
        raise ValueError(f"Unknown data generation type: {cfg.data_gen_type}")
    
    return torch.cat([batch, negatives], dim=0)

@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg: DataGenConfig) -> None:
    logger.info("Data Generation Config:\n%s", OmegaConf.to_yaml(cfg))
    
    # Load data using the function from data.py.
    train_loader, _, _ = load_and_split_data(cfg)
    
    # Get one batch from the training loader.
    batch = next(iter(train_loader))
    # If the DataLoader returns a tuple (data, labels), extract the data.
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch
    
    logger.info("Original batch shape: %s", images.shape)
    
    new_images = generate_data(images, cfg)
    logger.info("New batch shape after data generation: %s", new_images.shape)
    
    # Optionally, log the new batch shape to wandb.
    wandb.log({"new_batch_shape": new_images.shape})
    
    # Finish the wandb run if it was initiated.
    wandb.finish()

if __name__ == "__main__":
    main()
