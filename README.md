
# NanoGPT  

This repository contains two main files: `train.py` and `model.py`, which together implement and train a GPT model using PyTorch. The script supports both single-GPU and Distributed Data Parallel (DDP) training for multi-GPU setups.

## Files Overview
- **`train.py`**: This file is responsible for managing the training loop, model evaluation, and distributed training setup.
- **`model.py`**: Defines the architecture and configuration of the GPT model, including self-attention layers, feed-forward networks, and utility functions for autoregressive text generation.

## install

```
pip install torch numpy wandb 
```

Dependencies:

- [pytorch](https://pytorch.org) 
- [numpy](https://numpy.org/install/) 
-  `wandb` as a tool for logging and visualizing model training metrics.

## 1. Quick Start
The fastest way to get started is to train a character-level GPT on the works of Shakespeare. 
First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:
```sh
sh data/shakespeare_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. 
Now it is time to train the GPT. 
The size of it very much depends on the computational resources of your system:


**I have a GPU**. We can quickly train a baby GPT with the settings provided in the [config/train_shakespeare_char.py](config/train_shakespeare_char.py) config file:

```sh
sh train.py config/train_shakespeare_char.py
```

Inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. 
On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. 
Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-shakespeare-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
sh sample.py --out_dir=out-shakespeare-char
```

**I only no GPU** (or only regular computer/laptop). 
No worries, we can still train a GPT. 
Get PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. 
But even without it, a simple train run could look as follows:

```sh
sh train.py config/train_shakespeare_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. 
Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. 
We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). 
Because our network is so small we also ease down on regularization (`--dropout=0.0`). 
This still runs in about ~3 minutes, but gets a little bit more loss and therefore also worse samples, but it's still good fun:

```sh
sh sample.py --out_dir=out-shakespeare-char --device=cpu
```

## 2. Key Imports
The script imports several modules, including:
- **Standard Python Modules**: `os`, `time`, `math`, `pickle`, `contextlib.nullcontext`
- **Numerical Operations**: `numpy (np)`
- **PyTorch**: The main deep learning library.
- **Distributed Training**: `torch.nn.parallel.DistributedDataParallel (DDP)` and `torch.distributed` functions.

## 3. Suppressing Compilation Errors
The following code [add] disables specific Torch compilation errors to avoid crashes:
```sh
import torch._dynamo
torch._dynamo.config.suppress_errors = True
```

## 4. Model Configuration and Initialization
```sh
from model import GPTConfig, GPT
```
These classes are custom-defined in a separate `model.py` file, which define the GPT model configuration and architecture.

## 5. Configuration Variables
The script sets up default values for various parameters, such as:
- `out_dir`: Directory to save model checkpoints and logs.
- `eval_interval`: Number of iterations before running evaluation.
- `eval_only`: If `True`, the script exits after the first evaluation.
- `init_from`: Specifies whether to initialize the model from scratch or resume from a checkpoint.

## 6. Training and Model Parameters
- **WandB Logging**: Disabled by default. Can be set up for tracking training metrics.
in **`train.py`** WandB logging is controlled by the configuration flags in the `train.py` file. By default, it is disabled.
```sh
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
```
These lines initialize the logging system but keep it disabled (wandb_log = False). 
If you enable WandB logging by setting `wandb_log = True`, then you will need to initialize **WandB** within the training loop to start tracking metrics.

- **Gradient Accumulation**: Controls how many gradient steps are accumulated before updating the model parameters.
in **`train.py`** Gradient accumulation is implemented in the training loop. The part of the code is:
```sh
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
```
This sets the number of steps after which gradients will be accumulated and used for an optimizer update.
Inside the training loop:
```sh
for micro_step in range(gradient_accumulation_steps):
    # forward pass
    with ctx: 
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
    # backward pass, with gradient scaling if training in fp16
    scaler.scale(loss).backward()
```

- The loss is divided by `gradient_accumulation_steps` so that the accumulated gradients are averaged out.
- The backward pass `(scaler.scale(loss).backward())` is called inside the loop, but the optimizer update happens after the accumulation steps are completed.
So, the **gradient accumulation** happens within this loop, where the loss is scaled down and gradients are accumulated over several steps before the optimizer updates the model parameters.

## 7. Distributed Data Parallel Setup
appears in **`train.py`**, This code is used to check whether the script is running in Distributed Data Parallel (DDP) mode, which is a method to train a model on multiple GPUs across one or more machines.
```sh
ddp = int(os.getenv('LOCAL_RANK', -1)) != -1
```

```sh
os.getenv('LOCAL_RANK', -1):
```
This retrieves the environment variable LOCAL_RANK. When DDP is used, PyTorch typically assigns a local rank to each GPU, which is stored in this environment variable.
If the environment variable LOCAL_RANK is not set (i.e., the script is running in non-DDP mode), the function will return -1 as the default value.

```sh
int(os.getenv('LOCAL_RANK', -1)) != -1:
```
This checks if the LOCAL_RANK is not -1. If it is not -1, that means the script is running in DDP mode. If LOCAL_RANK is -1, it indicates the script is running on a single GPU or CPU, not in a distributed setting.
The variable ddp becomes True if the script is running in DDP mode and False if not.

If DDP is enabled, the script initializes the process group for multi-GPU training.
What Happens if DDP is Enabled?
If DDP mode is enabled:
- The script will split the work (batches of data) across multiple GPUs.
- Each GPU (or device) will independently compute its portion of the model's gradients.
- The gradients are then synchronized across all GPUs so that the model updates are consistent.

If **DDP mode** is not enabled, the model will train on a single device (e.g., one GPU or the CPU), and no distributed process group will be created.

## 8. Model Loading

These piece of code in the `train.py` file is responsible for defining and initializing the GPT model by specifying its architecture using certain parameters. 
```sh
model_args = dict(n_layer=12, n_head=12, n_embd=768)
```
Note:
- `n_layer=12`: This specifies the number of transformer layers (blocks) in the GPT model. 
In this case, the GPT model will have 12 layers. Each layer typically consists of self-attention and feed-forward sub-layers.

- `n_head=12`: This specifies the number of attention heads in each multi-head attention mechanism. 
Multi-head attention allows the model to focus on different parts of the input sequence simultaneously. 
Here, the model will have 12 attention heads for each layer.

- `n_embd=768`: This specifies the size of the embedding vector (the dimensionality of the token embeddings). 
The embedding size represents how each token is encoded into a fixed-size vector. 
In this case, each token is represented as a 768-dimensional vector.


```sh
model = GPT(GPTConfig(**model_args))
```
Note:
- `GPTConfig`: This is a class defined in the `model.py` file that holds the configuration for the GPT model. 
It takes various parameters like n_layer, n_head, and n_embd, and passes them to the model's architecture.

- `GPT`: This is the main GPT model class that is also defined in `model.py`. 
It builds the full architecture of the GPT model based on the configuration provided.

- `GPTConfig(**model_args)`: The `**model_args` syntax unpacks the dictionary model_args and passes each key-value pair as arguments to the `GPTConfig` constructor. 
So, effectively, the `GPTConfig` constructor is called with the following arguments:
```sh
GPTConfig(n_layer=12, n_head=12, n_embd=768)
```
This configures the model to have 12 layers, 12 attention heads, and an embedding size of 768.

What Happens Here?
- `GPTConfig(**model_args)`: This creates a configuration object that stores the parameters (n_layer, n_head, n_embd) needed to build the GPT model.
- `GPT(GPTConfig(...))`: The GPT class then takes the configuration object as input and constructs the GPT model according to the specified architecture (12 layers, 12 attention heads, 768-dimensional embeddings). 
Internally, the GPT class will create a series of self-attention layers, feed-forward networks, and other components that define the full model.


## 9. Training Loop
This code snippet appears to be from the `train.py` file and is responsible for executing the training process over a set number of iterations.
The script contains a main training loop that handles data loading, forward pass, backpropagation, and loss scaling:
```sh
for iter_num in range(max_iters):
    X, Y = get_batch('train')
    with ctx: # mixed precision training
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    scaler.scale(loss).backward()
```

#**A. Training Loop Initialization**
```sh
for iter_num in range(max_iters):
```
- `iter_num`: This variable keeps track of the current iteration number.
- `max_iters`: This variable is the total number of iterations the model will run for. 
It defines how long the training process will last.

The loop iterates over `max_iters`, which means the training continues until the specified number of iterations is completed.


#**B. Batch Loading**
```sh
X, Y = get_batch('train')
```
- `get_batch('train')`: This function fetches a batch of data from the training dataset.
  `X`: Represents the input data (e.g., token sequences) for the model to process.
  `Y`: Represents the corresponding target outputs or labels (e.g., the expected next token).
- `'train'`: Indicates that the batch is being drawn from the training dataset. 

 
#**C. Mixed Precision Training Context**
```sh
with ctx:
```
`ctx`: This likely refers to a context manager for **mixed-precision training**. Mixed precision training allows for faster computations by using **half-precision (FP16)** where appropriate, while still maintaining the accuracy of calculations by using full precision (FP32) for critical operations. 
This is a common technique to speed up training and reduce memory usage, especially when training on GPUs.


#**D. Forward Pass (Model Inference)**
```sh
logits, loss = model(X, Y):
```

`model(X, Y)`: The forward pass of the model is computed here.
  `X`: Input data (e.g., sequences of tokens).
  `Y`: The target labels (e.g., the expected next token in each sequence).
  `logits`: These are the raw predictions made by the model (before applying a softmax function). Logits represent the model’s confidence about different possible outputs.
  `loss`: This is the calculated loss for the current batch, which measures how far off the model's predictions (logits) are from the actual target values (Y).


#**E. Loss Scaling for Gradient Accumulation**
```sh
loss = loss / gradient_accumulation_steps
```
**Gradient Accumulation**: Instead of updating the model parameters after every batch, the script accumulates gradients over multiple batches before performing an update. This is done to simulate training with a larger batch size, which is often necessary when the available memory is limited.
  `gradient_accumulation_steps`: The number of steps (batches) over which gradients will be accumulated before performing a model update.
**Scaling the Loss**: The loss is divided by 
   `gradient_accumulation_steps` to ensure that when the gradients are accumulated over multiple batches, the overall scale of the gradients remains correct. This avoids artificially inflating the gradients due to multiple backward passes before an update.


#**F. Backward Pass (Backpropagation)**
```sh
scaler.scale(loss).backward()
```
- `scaler.scale(loss)`: This refers to automatic mixed-precision (AMP) scaling, which is used in mixed-precision training to avoid issues with small gradients when using FP16 precision.
  **Gradient Scaling**: Before the backward pass, the loss is scaled up by a factor to prevent the gradients from becoming too small to represent with half-precision floats. The `scaler.scale()` function scales the loss before calculating gradients.
- `.backward()`: This computes the **backward pass**, or **backpropagation**, which calculates the gradients of the loss with respect to each of the model's parameters. These gradients will later be used to update the model parameters.

To conclude,
Here is the breakdown of what happens in this training loop:
- **Data Loading**: A batch of training data (`X`, `Y`) is fetched using `get_batch('train')`.
- **Mixed Precision Context**: The model operates in a mixed-precision mode inside the `with ctx`: block, where certain calculations use lower precision for faster computations and memory savings.
- **Forward Pass**: The model computes predictions (`logits`) and the loss by processing the input batch (`X`, `Y`).
- **Loss Scaling for Gradient Accumulation**: The loss is divided by `gradient_accumulation_steps` to ensure gradients are accumulated across multiple batches and then averaged.
- **Backward Pass**: The scaled loss is backpropagated through the model to compute gradients for each parameter, which will eventually be used to update the model parameters.

## 10. Gradient Clipping and Optimization
#**A. Gradient Clipping**
```sh
if grad_clip != 0.0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

Gradient clipping is a technique used to prevent the issue of exploding gradients during the backpropagation process. 
**Exploding gradients** can occur when the gradients become excessively large, causing the model parameters to update too drastically and destabilize the training process.
Clipping the gradients ensures that they are capped at a certain threshold, thus keeping the model's updates more stable.
- `grad_clip`: This is a threshold value, and if it's set to `0.0`, gradient clipping is disabled. Otherwise, gradients will be clipped if they exceed this value.
- `torch.nn.utils.clip_grad_norm_()`: This function clips the gradients of the model’s parameters to ensure their norm (magnitude) does not exceed the specified threshold `(grad_clip)`.
- `model.parameters()`: Refers to the set of all parameters in the model that require gradients. These are the parameters for which the gradients were computed during backpropagation.
- `clip_grad_norm_()`: Computes the norm (magnitude) of the gradients, and if this value exceeds `grad_clip`, the gradients are scaled down proportionally so that their norm equals `grad_clip`. This prevents very large gradient updates from destabilizing the model's training process.

**Without Gradient Clipping:**
Large gradients can cause very large parameter updates, which may lead to instability in training (such as divergence of the loss or overshooting optimal values).

**With Gradient Clipping:**
The gradients are capped at a specific norm, keeping the updates more controlled and stable.

#**B. Optimizer Step with Mixed-Precision (Automatic Mixed Precision - AMP)**
```sh
scaler.step(optimizer)
```

This line performs the **optimizer step**, which updates the model's parameters based on the gradients that were computed during the backward pass. In this case, the update is scaled using **Automatic Mixed Precision (AMP)**, which enables faster training by using half-precision where appropriate.
- `scaler.step(optimizer)`: This is part of PyTorch's **mixed-precision training mechanism**. After the gradients have been scaled (using `scaler.scale()` during backpropagation), the `scaler.step()` function unscales the gradients and applies them to the optimizer.
- The optimizer (such as Adam or SGD) uses these gradients to adjust the model's parameters, which minimizes the loss and improves the model’s performance over time.
- **Mixed precision** allows for faster training by using half-precision (FP16) for some operations and full precision (FP32) for others. This saves memory and increases throughput on GPUs.

#**C. Updating the Scaler (For Mixed Precision)**
After each optimizer step, the scaler is updated to adjust the scaling factor used for the next iteration in mixed-precision training.

```sh
scaler.update()
```

`scaler.update()`: This function dynamically adjusts the scaling factor for the loss based on whether there were any **overflowing gradients** in the current step. Overflowing gradients occur when the gradient values are too large to be represented in FP16, and the scaler will reduce the scaling factor to prevent future overflows.
  - If the gradients did not overflow, the scaler might increase the scaling factor to maximize the benefits of mixed precision.
  - This **dynamic scaling** process ensures that the loss and gradients are scaled appropriately during mixed-precision training, which balances stability and performance.

To conclude:
- **Gradient Clipping:**
  - Before updating the model's parameters, the script checks if gradient clipping is enabled (i.e., `grad_clip != 0.0`).
  - If enabled, the gradients are clipped to a specific threshold to prevent large updates that could destabilize the model. This helps prevent exploding gradients during training.
- **Optimizer Step (Mixed Precision):**
  - After the gradients are clipped, `scaler.step(optimizer)` is called to perform the optimizer step in a mixed-precision setting.
  - The optimizer (e.g., Adam) updates the model's parameters using the gradients that were computed during backpropagation.
- **Scaler Update (Mixed Precision):**
  - The `scaler.update()` function adjusts the scaling factor for the next iteration based on whether any gradients overflowed. This ensures the training process remains stable by using the appropriate scaling factor in future steps.


## 11. Logging
Logs are printed after a set number of iterations to track the loss of the *training loop*:

#**A. Logging Interval**
```sh
if iter_num % log_interval == 0:
```

- `iter_num`: This is the current iteration number in the training loop.
- `log_interval`: This is a predefined configuration parameter that specifies how often the script should log information. 
For example, if log_interval = 100, then the script logs information every 100 iterations.
This line checks whether the current iteration number `(iter_num)` is a multiple of `log_interval`. If true, the logging process will be triggered. The goal here is to avoid logging every single iteration, which would produce too much output. Instead, logging happens after a set number of iterations to provide periodic updates on the training progress.


#**B. Extracting the Loss Value**
```sh
    lossf = loss.item() * gradient_accumulation_steps
    print(f"iter {iter_num}: loss {lossf:.4f}")
```
- `loss`: This is the loss tensor computed during the forward pass of the model. The loss is used to measure how far off the model's predictions are from the actual targets (labels).
- `loss.item()`: The `.item()` method is used to extract the scalar value from the loss tensor (since loss is a PyTorch tensor, you need to convert it to a Python float for easy logging and display).
- `gradient_accumulation_steps`: This multiplication reverses the scaling applied to the loss during gradient accumulation. Since the loss was divided by gradient_accumulation_steps earlier in the training loop (to account for gradient accumulation), it is multiplied by the same value here to get the original, unscaled loss value.

also, this line outputs the loss to the console using Python’s f-string for formatting. 


## 12. Distributed Training Cleanup
At the end of training, the process group is destroyed if distributed training was used:
```sh
if ddp:
    destroy_process_group()
```