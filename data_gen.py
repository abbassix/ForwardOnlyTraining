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

# Import the data loading function and base configuration from data.py
from data import load_and_split_data, Config as DataConfig

logger = logging.getLogger(__name__)

@dataclass
class DataGenConfig(DataConfig):
    # Specify the data generation type. Default is "random_noise".
    data_gen_type: str = "random_noise"

def generate_data(batch: torch.Tensor, cfg: DataGenConfig) -> torch.Tensor:
    """
    Generate new data based on the specified method in the config.
    
    For "random_noise": Given a batch of shape (B, C, W, H),
    generate B samples of white noise with the same shape and concatenate them,
    resulting in an output batch of shape (2*B, C, W, H).
    """
    if cfg.data_gen_type == "random_noise":
        noise = torch.randn_like(batch)
        augmented_batch = torch.cat([batch, noise], dim=0)
        return augmented_batch
    else:
        raise ValueError(f"Unknown data generation type: {cfg.data_gen_type}")

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
