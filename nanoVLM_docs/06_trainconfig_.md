# Chapter 6: TrainConfig

Welcome back to the nanoVLM tutorial! We've now spent several chapters building up our understanding of the [VisionLanguageModel (VLM)](05_visionlanguagemodel__vlm__.md): we explored its foundational blueprint in the [VLMConfig](01_vlmconfig_.md), met its core components ([Vision Transformer (ViT)](02_vision_transformer__vit__.md), [Modality Projector (MP)](03_modality_projector__mp__.md), and [Language Model (LM)](04_language_model__lm__.md)), and saw how the main [VisionLanguageModel](05_vision_language_model__vlm__.md) class brings them all together.

So, we have the complete model built and ready. What's next? We need to **train** it!

## The "Cooking Instructions": What is TrainConfig?

Imagine you've just assembled a brand new, incredibly complex machine (that's our [VisionLanguageModel](05_vision_language_model__vlm__.md)). It's built according to its blueprint ([VLMConfig](01_vlmconfig_.md)) and has all its sophisticated parts ([ViT](02_vision_transformer__vit__.md), [MP](03_modality_projector__mp__.md), [LM](04_language_model__lm__.md)). But right now, it doesn't *know* how to perform Vision-Language tasks. It needs to be taught using data.

Training is the process of feeding the model lots of examples (image-text pairs) and adjusting its internal settings (its weights) so it gets better at generating the correct text output for a given image and prompt.

This training process itself needs a set of instructions, a plan for *how* the learning should happen. This plan is stored in the **`TrainConfig`**.

Think of the `TrainConfig` as the **"cooking instructions"** for training the VLM. It holds all the specific settings, often called **hyperparameters**, that control the training loop. These settings tell the training process things like:

*   How fast should the model learn? (Learning Rate)
*   How many examples should it look at simultaneously? (Batch Size)
*   How many times should it go through the entire dataset? (Epochs)
*   Where can it find the training data? (Dataset Paths)
*   Should we save progress reports? (Logging Options)

Just like you use the [VLMConfig](01_vlmconfig_.md) to create the model itself, you use the `TrainConfig` to define *how* that model will be trained. It's the second crucial configuration file in nanoVLM.

## Using TrainConfig to Train the Model

The primary place where the `TrainConfig` is used is in the `train.py` script. This script contains the main `train` function that orchestrates the entire training process.

First, you need to import `TrainConfig`:

```python
# From train.py or models/config.py
from models.config import TrainConfig
```

Then, you create an instance of `TrainConfig`. Like `VLMConfig`, it comes with sensible default values.

```python
# Create a TrainConfig instance with default settings
train_cfg = TrainConfig()

print(train_cfg)
```

If you print `train_cfg`, you'll see all the default training hyperparameters, like `batch_size=256`, `epochs=5`, `lr_mp=2e-3`, etc.

Often, you'll want to adjust these defaults based on your available hardware (like GPU memory), the specific dataset you're using, or your experimental goals. You can override defaults when creating the instance:

```python
# Create a TrainConfig instance, overriding the default batch size and epochs
custom_train_cfg = TrainConfig(batch_size=128, epochs=10)

print(custom_train_cfg.batch_size) # Output: 128
print(custom_train_cfg.epochs)    # Output: 10
```

In `train.py`, the `main` function parses command-line arguments and uses them to potentially override the default settings in both `VLMConfig` and `TrainConfig` before passing them to the core `train` function.

```python
# Simplified from train.py main()
def main():
    parser = argparse.ArgumentParser()
    # Add arguments like --lr_mp, --batch_size, etc.
    parser.add_argument('--lr_mp', type=float, help='Learning rate for MP')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    # ... other args ...
    args = parser.parse_args()

    # Create default configs
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig() # Default TrainConfig

    # Override train_cfg settings based on command line args
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    # ... override other settings ...

    print("--- Train Config ---")
    print(train_cfg) # Print the final config being used

    # Pass BOTH configs to the main training function
    train(train_cfg, vlm_cfg)
```

The `train` function then receives this `train_cfg` object (and the `vlm_cfg`) and uses its values throughout the training loop.

```python
# Simplified from train.py train() function signature
def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    # train_cfg now holds all the settings for THIS training run
    # ... rest of training setup and loop ...
    pass
```

This pattern keeps the training logic clean, as all the knobs and dials for training are neatly organized within the `train_cfg` object.

## Key Settings in TrainConfig

Let's look at some of the important hyperparameters defined in `TrainConfig` and what they control:

*   **`lr_mp` and `lr_backbones`:** Learning Rates. These are arguably the most important settings. They determine how large of a step the optimizer takes when adjusting the model's weights based on the calculated loss. A higher learning rate means faster, potentially unstable learning; a lower rate means slower, potentially more stable learning. Notice that nanoVLM uses *two* different learning rates: one for the [Modality Projector (MP)](03_modality_projector__mp__.md) (`lr_mp`) and a possibly different, usually smaller one, for the [ViT](02_vision_transformer__vit__.md) and [LM](04_language_model__lm__.md) backbones (`lr_backbones`). This is because the MP is usually initialized randomly and needs to learn its job relatively quickly, while the backbones are often pre-trained and only need fine-tuning.
*   **`batch_size`:** How many image-text pairs are processed together in one forward and backward pass. Larger batch sizes can lead to more stable gradient estimates but require significantly more GPU memory.
*   **`epochs`:** One epoch means the training process has gone through the *entire* training dataset once. `epochs` sets the total number of times this repetition happens. More epochs often lead to better performance but take longer.
*   **`train_dataset_path`, `train_dataset_name`, `test_dataset_path`:** These strings specify where the data is located (e.g., a path to a directory or a dataset name on the Hugging Face Hub). `train_dataset_name` is a tuple because you can train on multiple datasets combined (as seen in the code snippet).
*   **`val_ratio`:** The fraction of the training data to set aside for validation (checking the model's performance during training on data it hasn't strictly trained on).
*   **`eval_in_epochs`:** A boolean flag to control whether evaluation (running the model on a separate test set like MMStar) happens periodically *during* the training epochs, or only at the very end.
*   **`compile`:** A boolean flag to enable `torch.compile`, which can significantly speed up training by optimizing the model's computational graph.
*   **`resume_from_vlm_checkpoint`:** A boolean flag indicating whether the training should start by loading weights from a previously saved full VLM checkpoint (specified by `vlm_checkpoint_path` in `VLMConfig`), rather than just loading the individual backbone weights.
*   **`log_wandb` and `wandb_entity`:** Settings related to logging training metrics and results to Weights & Biases (WandB), a popular experiment tracking platform.

These are just some of the key settings. Adjusting these values is how you control the training experiment and try to find the best way to teach the VLM.

## Under the Hood: How TrainConfig Directs Training

Just like `VLMConfig`, the `TrainConfig` is a `dataclass`. It simply holds the configuration values. The actual logic that *uses* these values lives within the training script, primarily in the `train` function in `train.py`.

Here's a simplified sequence diagram showing how the `TrainConfig` flows through the training process:

```mermaid
sequenceDiagram
    participant User as User/CLI
    participant ConfigFile as models/config.py
    participant TrainConfigClass as TrainConfig
    participant TrainScript as train.py
    participant DataLoaderModule as DataLoader
    participant OptimizerModule as Optimizer
    participant VLM as VisionLanguageModel
    participant Wandb as WandB

    User->>TrainScript: Run train.py (maybe with args)
    TrainScript->>TrainConfigClass: Create TrainConfig()<br>(Apply overrides from args)
    TrainConfigClass-->>TrainScript: train_cfg object

    TrainScript->>DataLoaderModule: Create DataLoader(train_cfg, vlm_cfg, ...)
    DataLoaderModule->>train_cfg: Read train_cfg.batch_size,<br>train_cfg.train_dataset_path,<br>train_cfg.val_ratio, ...
    DataLoaderModule-->>TrainScript: train_loader, val_loader

    TrainScript->>VLM: Create/Load VLM(vlm_cfg)
    VLM-->>TrainScript: model instance

    TrainScript->>OptimizerModule: Create Optimizer(model.parameters(), train_cfg)
    OptimizerModule->>train_cfg: Read train_cfg.lr_mp,<br>train_cfg.lr_backbones
    OptimizerModule-->>TrainScript: optimizer instance

    opt group Training Loop; Loop over train_cfg.epochs times
        TrainScript->>TrainScript: Loop over batches in train_loader (batch_size is set by loader)
        TrainScript->>VLM: model(batch)
        VLM-->>TrainScript: loss
        TrainScript->>VLM: loss.backward()
        TrainScript->>OptimizerModule: optimizer.step()
        alt If train_cfg.eval_in_epochs is True
            TrainScript->>VLM: model.eval()
            TrainScript->>TrainScript: Run validation/testing loop (uses train_cfg.mmstar_batch_size)
            TrainScript->>VLM: model.train()
        end
        alt If train_cfg.log_wandb is True
            TrainScript->>Wandb: Log metrics (loss, accuracy, etc.)
        end
    end

    TrainScript-->>User: Training complete, Final results

```

The `train_cfg` object is created once at the beginning and then passed around to different parts of the training setup:

1.  **Data Loading:** The `get_dataloaders` function reads `train_cfg.train_dataset_path`, `train_cfg.train_dataset_name`, `train_cfg.val_ratio`, `train_cfg.batch_size`, and `train_cfg.mmstar_batch_size` to configure how the data is loaded and batched.
2.  **Optimizer:** The `optim.AdamW` optimizer is initialized with different parameter groups (for MP and backbones), using `train_cfg.lr_mp` and `train_cfg.lr_backbones` for their respective learning rates.
3.  **Training Loop:** The main loop iterates `train_cfg.epochs` times. Inside the loop, batches are processed using the `batch_size` configured in the dataloader (which got it from `train_cfg`). Evaluation frequency is controlled by `train_cfg.eval_in_epochs`.
4.  **Logging:** If `train_cfg.log_wandb` is true, the script initializes WandB logging using `train_cfg.wandb_entity` and sends metrics like loss and accuracy to WandB based on the training progress.

Let's look at a few simplified code snippets from `train.py` that demonstrate this usage.

### Getting Data Loaders

```python
# Simplified from train.py get_dataloaders(...)
def get_dataloaders(train_cfg, vlm_cfg):
    # ... tokenizer and image_processor setup ...

    # Load datasets using paths/names from train_cfg
    train_ds = load_dataset(train_cfg.train_dataset_path, train_cfg.train_dataset_name)
    # ... concatenate, shuffle, and split using train_cfg.val_ratio ...

    # Create DataLoaders using batch_size from train_cfg
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size, # Used here
        shuffle=True,
        # ... other DataLoader args ...
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size, # Used here
        shuffle=False,
        # ... other DataLoader args ...
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.mmstar_batch_size, # Used here for test set
        shuffle=False,
        # ... other DataLoader args ...
    )
    return train_loader, val_loader, test_loader
```

The `batch_size`, dataset paths, and validation split ratio are all read directly from the `train_cfg` object to set up the data pipelines.

### Setting up the Optimizer

```python
# Simplified from train.py train() function
def train(train_cfg, vlm_cfg):
    # ... get dataloaders, initialize model ...

    # Define parameter groups with different learning rates from train_cfg
    param_groups = [{'params': model.MP.parameters(), 'lr': train_cfg.lr_mp}, # LR for MP
                    {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}] # LR for backbones
    
    optimizer = optim.AdamW(param_groups)

    # ... rest of training loop ...
```

Here, `train_cfg.lr_mp` and `train_cfg.lr_backbones` are used to tell the optimizer how aggressively to update the weights of the Modality Projector versus the vision and language backbones.

### The Training Loop

```python
# Simplified from train.py train() function
def train(train_cfg, vlm_cfg):
    # ... setup code ...

    # Loop for the specified number of epochs
    for epoch in range(train_cfg.epochs): # train_cfg.epochs used here
        model.train()
        total_train_loss = 0

        # Iterate through batches (size determined by train_cfg.batch_size in DataLoader)
        for batch in train_loader:
            # ... data to device ...
            
            optimizer.zero_grad()

            # Forward pass (VisionLanguageModel uses VLMConfig internally)
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            loss.backward()
            
            # Adjust learning rates based on step (uses train_cfg.lr_mp, train_cfg.lr_backbones)
            # ... get_lr(global_step, train_cfg.lr_mp, total_steps) ...
            optimizer.param_groups[0]['lr'] = adj_lr_mp
            optimizer.param_groups[1]['lr'] = adj_lr_backbones

            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Periodic evaluation during training
            if train_cfg.eval_in_epochs and global_step % 250 == 0: # train_cfg.eval_in_epochs used here
                # ... call evaluation function ...
                pass
            
            # Logging
            if train_cfg.log_wandb: # train_cfg.log_wandb used here
                # ... log metrics ...
                pass

            global_step += 1

        # ... end of epoch calculations and logging ...
```

The main training loop structure (number of epochs, when to evaluate, when to log) is directly controlled by values read from `train_cfg`.

## Under the Hood: `TrainConfig` Definition

Finally, let's look at the definition of the `TrainConfig` `dataclass` in `models/config.py`.

```python
# From models/config.py
from dataclasses import dataclass

@dataclass # This decorator makes it a dataclass
class TrainConfig:
    # Training hyperparameters with types and default values
    lr_mp: float = 2e-3           # Learning rate for Modality Projector
    lr_backbones: float = 1e-4    # Learning rate for ViT and LM backbones
    data_cutoff_idx: int = None   # Use only first N samples of the dataset
    val_ratio: float = 0.01       # Percentage of data for validation
    batch_size: int = 256         # Batch size for main training
    mmstar_batch_size: int = 32   # Batch size specifically for MMStar evaluation
    eval_in_epochs: bool = True   # Whether to run evaluation during training
    epochs: int = 5               # Total number of training epochs
    compile: bool = True          # Enable torch.compile
    resume_from_vlm_checkpoint: bool = False # Start from a saved VLM checkpoint
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron' # HF Hub path for training data
    train_dataset_name: tuple[str, ...] = ("ai2d", "aokvqa", "chart2text", ...) # Names of specific datasets to use
    test_dataset_path: str = "Lin-Chen/MMStar" # HF Hub path for test data
    wandb_entity: str = "HuggingFace" # WandB entity name
    log_wandb: bool = True        # Enable WandB logging
```

This definition simply lists all the configurable training parameters with their data types and default values. The `train.py` script then imports this class and uses instances of it to get these values and drive the training process.

## Conclusion

In this chapter, we've introduced the `TrainConfig`, the essential configuration object that provides the "cooking instructions" for training the [VisionLanguageModel (VLM)](05_vision_language_model__vlm__.md). We saw how it holds crucial hyperparameters like learning rates, batch size, epochs, and dataset paths. We learned how `train.py` uses the `TrainConfig` instance to set up the data loaders, configure the optimizer, control the training loop, and manage logging. Understanding `TrainConfig` is key to being able to run or customize the training process in nanoVLM.

While `TrainConfig` tells us *which* datasets to use and *how* to batch them, it doesn't detail the specifics of how the data is loaded, processed, and prepared for the model. That's the subject of our next chapter.

[Next Chapter: Data Handling (Datasets, Collators, Processors)](07_data_handling__datasets__collators__processors__.md)

