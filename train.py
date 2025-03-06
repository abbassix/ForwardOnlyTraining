"""
train.py

Trains each block sequentially by reusing the training logic from train_block.py.
Data augmentation occurs immediately before the currently trained block.
"""

import logging
import torch
import wandb
import hydra
from omegaconf import OmegaConf
from data import load_and_split_data
from model_factory import build_model
from train_block import train_single_block

logger = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg):
    logger.info("Training configuration:\n%s", OmegaConf.to_yaml(cfg))

    wandb.init(project=cfg.project, config=OmegaConf.to_container(cfg, resolve=True), reinit=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_loader, _, _ = load_and_split_data(cfg)
    model = build_model(cfg).to(device)

    for block_idx in range(len(model.blocks)):
        logger.info(f"Training block {block_idx + 1}/{len(model.blocks)}")

        # Train the current block using existing logic in train_block.py
        train_single_block(
            model=model,
            train_loader=train_loader,
            cfg=cfg,
            block_index=block_idx,
            device=device
        )

    wandb.finish()

if __name__ == "__main__":
    main()
