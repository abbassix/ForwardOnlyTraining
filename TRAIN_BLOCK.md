# Train Block Module Documentation

## Overview
This module (&#96;train_block.py&#96;) is responsible for training a model by loading data, generating augmented data, and updating model parameters based on binary classification loss. The module performs the following key tasks:
- Loads data using the &#96;load_and_split_data&#96; function.
- Generates augmented data by concatenating the original batch (positive samples) with random noise (negative samples).
- Passes the data through a model (e.g., &#96;PoolConv&#96; from &#96;models.py&#96;) and computes the loss using &#96;nn.BCEWithLogitsLoss&#96;.
- Updates model parameters using an optimizer (SGD or Adam).
- Logs training metrics using wandb.

## Training Process

### Data Loading and Augmentation
- **Data Loading:**  
  The module loads training data via the &#96;load_and_split_data&#96; function from &#96;data.py&#96;.
- **Data Augmentation:**  
  For each batch:
  - The original batch is used as positive samples.
  - A batch of random noise, generated using the &#96;generate_data&#96; function, is used as negative samples.
  - The two batches are concatenated, resulting in a new batch with a size of &#96;2 * batch_size&#96;.

### Model Forward Pass and Loss Computation
- **Forward Pass:**  
  The concatenated data is passed through the model to obtain the outputs.
- **Logits Calculation:**  
  1. The outputs are flattened and squared.
  2. The sum of squares (goodness) is computed.
  3. A threshold (equal to the flattened dimension) is subtracted from the goodness to obtain the logits.
- **Loss Labels:**  
  Binary labels are created where the first half (positive samples) are set to 1 and the second half (negative samples) to 0.
- **Loss Function:**  
  The module uses &#96;nn.BCEWithLogitsLoss&#96; to compute the loss between the logits and the binary labels.

### Optimization
- **Parameter Update:**  
  The model parameters are updated using an optimizer (either SGD or Adam) based on the computed loss.
- **Logging:**  
  Training metrics such as batch loss and epoch loss are logged using wandb.

## Configuration Parameters
The module uses Hydra for configuration management. Key parameters include:
- **epochs (int):** Number of training epochs.
- **lr (float):** Learning rate for the optimizer.
- **optimizer (str):** Optimizer type to use (&#96;adam&#96; or &#96;sgd&#96;).
- **data_gen_type (str):** Method for data generation (default: &#96;random_noise&#96;).
- **Model Parameters:**
  - **c_in (int):** Number of input channels (adjusted automatically based on the dataset).
  - **m (int):** Pooling kernel size; if set to 1, pooling is skipped.
  - **c_out (int):** Number of output channels.
  - **k (int):** Convolution kernel size.

## Usage Example
To run the training block, simply execute the module from the command line:

```python
python train_block.py
```
