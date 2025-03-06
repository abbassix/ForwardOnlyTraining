"""
blocks.py

This module defines the architecture of the blocks.
It includes a placeholder model named PoolConv that applies an optional max-pooling operation followed by a convolution.
The PoolConv module takes the following arguments:
  - c_in: Number of input channels.
  - m: The kernel size for max-pooling (m x m). If m == 1, pooling is skipped.
  - c_out: The number of convolutional kernels (output channels).
  - k: The kernel size for convolution (k x k).

The module returns a PyTorch model.
"""

import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class PoolConv(nn.Module):
    def __init__(self, c_in: int, m: int, c_out: int, k: int):
        """
        Initialize the PoolConv module.

        Parameters:
            c_in (int): Number of input channels.
            m (int): The kernel size for max pooling (m x m). If m == 1, no pooling layer is applied.
            c_out (int): The number of convolutional kernels (output channels).
            k (int): The kernel size for convolution (k x k).
        """
        super().__init__()

        # Parameter validation
        if c_in <= 0:
            raise ValueError("c_in must be a positive integer.")
        if m < 1:
            raise ValueError("m must be an integer greater than or equal to 1.")
        if c_out <= 0:
            raise ValueError("c_out must be a positive integer.")
        if k <= 0:
            raise ValueError("k must be a positive integer.")

        # Use pooling only if m > 1
        self.use_pool = m > 1
        if self.use_pool:
            self.pool = nn.MaxPool2d(kernel_size=m)

        # Convolutional layer: input channels = c_in, output channels = n, kernel size = k x k
        self.conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=k, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PoolConv module.

        Parameters:
            x (torch.Tensor): Input tensor of shape (batch_size, c_in, H, W).

        Returns:
            torch.Tensor: Output tensor after applying optional max pooling and convolution.
        """
        if self.use_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x

# For quick debugging/testing. In production, use a dedicated test suite (e.g., pytest).
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing PoolConv module.")

    # Example: Create a PoolConv model with pooling (m > 1)
    model_with_pool = PoolConv(c_in=3, m=2, c_out=16, k=3)
    logger.info("Model with pooling:\n%s", model_with_pool)
    
    # Example: Create a PoolConv model without pooling (m = 1)
    model_no_pool = PoolConv(c_in=3, m=1, c_out=16, k=3)
    logger.info("Model without pooling:\n%s", model_no_pool)
    
    # Create a random input tensor with shape (batch_size, channels, height, width)
    x = torch.randn(1, 3, 32, 32)
    
    # Forward pass through the model with pooling
    output_with_pool = model_with_pool(x)
    logger.info("Output shape with pooling: %s", output_with_pool.shape)
    
    # Forward pass through the model without pooling
    output_no_pool = model_no_pool(x)
    logger.info("Output shape without pooling: %s", output_no_pool.shape)
