"""
model_factory.py

This module defines a ModularModel class that builds a modular neural network
from blocks specified in the configuration. Each block is defined as a tuple (m, n, k),
where:
  - m: pooling size (if m == 1, no pooling is applied),
  - n: number of filters (output channels),
  - k: convolution kernel size.

The model is built using an nn.ModuleList, so that each block can be trained
separately. The forward method accepts an optional layer_index:
  - If layer_index is provided, the input is run through all preceding blocks
    under torch.no_grad, then the block at that index is run normally.
  - If no index is given, the input is run through all blocks sequentially.

A factory function, build_model(cfg), is provided to construct the model,
determining the appropriate input channel size based on the dataset and
using a dummy input to validate dimensions.
"""

import logging
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from typing import Optional
import hydra

# Import the block from blocks.py (formerly models.py)
from blocks import PoolConv

logger = logging.getLogger(__name__)

class ModularModel(nn.Module):
    def __init__(self, input_channels: int, block_configs: list):
        """
        Constructs a modular model using blocks specified in block_configs.

        Parameters:
            input_channels (int): Number of input channels (e.g., 1 for MNIST, 3 for CIFAR).
            block_configs (list): List of tuples/lists [(m, n, k), ...] where:
                m (int): Pooling size (if 1, no pooling is applied).
                n (int): Number of filters (output channels).
                k (int): Convolution kernel size.
        """
        super(ModularModel, self).__init__()
        self.blocks = nn.ModuleList()
        
        current_channels = input_channels
        for idx, block_cfg in enumerate(block_configs):
            m, n, k = block_cfg  # Unpack block configuration
            # Create a block with the current number of input channels.
            block = PoolConv(c_in=current_channels, m=m, c_out=n, k=k)
            self.blocks.append(block)
            logger.info(f"Added block {idx}: PoolConv(c_in={current_channels}, m={m}, c_out={n}, k={k})")
            current_channels = n  # Update input channels for the next block

    def forward(self, x: torch.Tensor, layer_index: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass of the model.

        Parameters:
            x (torch.Tensor): Input tensor.
            layer_index (int, optional): If provided, runs the blocks before this index
                with torch.no_grad, then runs the block at layer_index normally and returns its output.
                If None, runs all blocks sequentially.

        Returns:
            torch.Tensor: The output tensor from the specified forward pass.
        """
        if layer_index is not None:
            # Run preceding blocks with no gradient computation.
            for i in range(layer_index):
                with torch.no_grad():
                    x = self.blocks[i](x)
            # Run the specified block normally (allow gradients) and return its output.
            x = self.blocks[layer_index](x)
            return x
        else:
            # Run through all blocks normally.
            for block in self.blocks:
                x = block(x)
            return x

def build_model(cfg) -> ModularModel:
    """
    Factory function to build a ModularModel based on configuration.

    The configuration must include:
      - dataset: used to determine the number of input channels.
      - model: a list of block configurations, each being a tuple/list (m, n, k).

    Parameters:
        cfg (dict or OmegaConf): Configuration object.

    Returns:
        ModularModel: The constructed modular model.
    """
    # Determine input channels based on dataset type.
    dataset = cfg.get("dataset", "mnist").lower()
    if dataset == "mnist":
        input_channels = 1
    else:
        input_channels = 3  # For CIFAR-10, CIFAR-100, etc.
    
    # Retrieve model block configuration from the config.
    block_configs = cfg.get("model", [])
    if not block_configs:
        raise ValueError(
            "No model block configuration provided in config. "
            "Please provide a 'model' section with block tuples [(m, n, k), ...]."
        )
    
    logger.info(f"Building ModularModel with input_channels={input_channels} and block_configs={block_configs}")
    model = ModularModel(input_channels=input_channels, block_configs=block_configs)
    
    # Validate model dimensions with a dummy input.
    dummy_input_size = (1, input_channels, 32, 32) if dataset != "mnist" else (1, input_channels, 28, 28)
    dummy_input = torch.randn(dummy_input_size)
    logger.info(f"Dummy input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    logger.info(f"Output shape after dummy input: {output.shape}")
    
    return model

if __name__ == "__main__":
    # This block is for testing and demonstration.
    # Create an example configuration. In practice, Hydra will supply the configuration.
    cfg_dict = {
        "dataset": "cifar10",
        "model": [
            [1, 8, 3],
            [2, 16, 3],
            [2, 34, 3]
        ]
    }
    cfg = OmegaConf.create(cfg_dict)
    model = build_model(cfg)
    # Run a dummy input through the full model.
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print("Final output shape:", output.shape)
