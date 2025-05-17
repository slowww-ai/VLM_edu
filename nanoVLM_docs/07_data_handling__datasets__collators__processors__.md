# Chapter 7: Data Handling (Datasets, Collators, Processors)

Welcome back to the nanoVLM tutorial! In our previous chapter, [Chapter 6: TrainConfig](06_trainconfig_.md), we learned about the configuration (`TrainConfig`) that tells us *how* to train our [VisionLanguageModel (VLM)](05_visionlanguagemodel__vlm__.md) – things like learning rates, batch size, and which datasets to use.

But how do we actually *get* the data (images and text) from files or online repositories, prepare it so the model can understand it (turn images into tensors, text into token IDs), and then group it efficiently for training?

This is the job of the data handling components: **Datasets**, **Processors**, and **Collators**.

## Preparing Ingredients for the Model

Imagine you're running a factory that builds complex machines (our VLM). Each machine needs a continuous supply of pre-processed parts (data batches) to learn and operate. You can't just throw raw materials (like a JPEG image file or a plain text string) into the machine. They need to be prepared first!

This preparation process has several steps:

1.  **Gathering Individual Parts:** You need to know where to find each piece of raw material (like finding a specific image file and its corresponding text caption or question). This is handled by **Datasets**.
2.  **Shaping the Parts:** The raw materials need to be shaped and measured into standard formats the machine can handle (like resizing an image to a fixed size or converting text into numerical IDs). This is handled by **Processors**.
3.  **Grouping and Arranging:** You need to gather several prepared parts together and arrange them neatly so the machine can process them efficiently in batches. This includes making sure all text sequences in a batch are the same length by adding padding. This is handled by **Collators**.

Datasets, Processors, and Collators work together in nanoVLM to create the perfectly formatted batches of data that the [VisionLanguageModel](05_vision_language_model__vlm__.md) needs for training and inference.

## Key Components

Let's break down the roles of these three types of components:

*   **Datasets:** Responsible for loading *individual* data samples (one image, one text pair) from their source (like a file or a dataset object from a library). They define what a single example looks like.
*   **Processors:** Responsible for converting raw data (like a PIL Image object or a string of text) into the numerical format that neural networks understand (like PyTorch tensors of pixels or token IDs).
*   **Collators:** Responsible for taking a *list* of these individual processed samples and combining them into a single *batch* of tensors. This is where things like padding sequences to a uniform length and creating labels for training happen.

Now let's look at how this works in the nanoVLM code.

## Datasets: Loading Individual Samples

In nanoVLM, Dataset classes inherit from PyTorch's `torch.utils.data.Dataset`. Their main job is to implement two methods:

*   `__len__`: Returns the total number of samples in the dataset.
*   `__getitem__`: Takes an index (a number) and returns one single sample (an image, its related text, etc.).

nanoVLM uses different dataset classes for different data formats or tasks, like `VQADataset` for general Visual Question Answering data (like datasets from Hugging Face's `the_cauldron`) and `MMStarDataset` for the specific format used in the MMStar benchmark.

Let's look at a simplified example of `VQADataset` from `data/datasets.py`.

```python
# From data/datasets.py
from torch.utils.data import Dataset
from PIL import Image # Used for image handling

class VQADataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor):
        # The raw dataset object (e.g., from Hugging Face datasets library)
        self.dataset = dataset 
        # Processors are often passed here so getitem can use them
        self.tokenizer = tokenizer 
        self.image_processor = image_processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load and process the image using the processor
        image_data = item['images'][0] # Get the first image if it's a list
        processed_image = self.image_processor(image_data.convert('RGB'))

        # Get and format the text data (question and answer)
        text = item['texts'][0] # Get the first text item
        question = text['user']
        answer = text['assistant'] + self.tokenizer.eos_token # Add end-of-sequence token

        formatted_text = f"Question: {question} Answer:"

        # Return the processed image and formatted text parts
        return {
            "image": processed_image, # This is a tensor now
            "text_data": formatted_text, # This is still a string
            "answer": answer # This is still a string
        }
```

The `__getitem__` method takes an index (`idx`), retrieves the raw item from the underlying dataset, uses the `image_processor` (which we'll see next) to turn the raw image into a tensor, formats the text, and returns a dictionary containing the processed image tensor and the raw text strings (question and answer).

Note that `MMStarDataset` is very similar, just tailored to the specific format of the MMStar evaluation dataset. It also uses the `image_processor` and `tokenizer` to get inputs ready.

## Processors: Converting Raw Data

Processors are the tools that convert raw inputs (like a `PIL.Image` object or a Python string) into numerical tensors that the model can process.

In nanoVLM, the `data/processors.py` file contains simple functions to get the necessary processors:

*   `get_image_processor`: Returns a function (specifically, a `torchvision.transforms.Compose` object) that resizes the image and converts it into a PyTorch tensor.
*   `get_tokenizer`: Returns a Hugging Face `transformers` tokenizer object, which converts text strings into sequences of numerical token IDs and vice-versa.

```python
# From data/processors.py
from transformers import AutoTokenizer
import torchvision.transforms as transforms

# Cache tokenizers to avoid reloading
TOKENIZERS_CACHE = {}

def get_tokenizer(name):
    if name not in TOKENIZERS_CACHE:
        # Load tokenizer by name (e.g., 'HuggingFaceTB/cosmo2-tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
        # Set padding token to be the same as end-of-sequence token
        tokenizer.pad_token = tokenizer.eos_token 
        TOKENIZERS_CACHE[name] = tokenizer
    return TOKENIZERS_CACHE[name]

def get_image_processor(img_size):
    # Returns a sequence of image transformations
    return transforms.Compose([
        # Resize the image to the required model input size
        transforms.Resize((img_size, img_size)), 
        # Convert PIL Image to PyTorch Tensor
        transforms.ToTensor() 
    ])
```

These processors are crucial because they ensure that all images are the same size and all text is represented as numbers, matching the inputs expected by the [Vision Transformer (ViT)](02_vision_transformer__vit__.md) and the [Language Model (LM)](04_language_model__lm__.md). As seen in the `VQADataset` snippet, the `image_processor` is used *within* the Dataset's `__getitem__`. The `tokenizer`, however, is primarily used by the **Collator** to process batches of text, and also in inference scripts ([generate.py](generate.py)) to encode/decode prompts and outputs.

## Collators: Batching and Padding

The Collator is where the magic of batching happens. A `torch.utils.data.DataLoader` takes a Dataset and a `batch_size`, gets a list of individual samples by calling `Dataset.__getitem__` multiple times, and then passes this list to the Collator's `__call__` method. The Collator's job is to take this list of dictionaries (like the ones returned by `VQADataset.__getitem__`) and combine them into a *single* dictionary of PyTorch tensors, where each tensor has an added batch dimension.

This is relatively straightforward for images (they are all the same size thanks to the `image_processor`, so you can just stack them). However, text sequences (questions and answers) will almost certainly have different lengths. Neural networks require tensors of uniform shape, so the Collator must handle **padding** – adding special "padding" tokens to the shorter sequences to make them all the same length as the longest sequence in the batch. It also needs to create an **attention mask** so the model knows which tokens are real and which are just padding.

Furthermore, for training, the Collator needs to prepare the **labels**. In nanoVLM's training setup, we train the model to predict the *answer* tokens given the combined image and question. This means the labels tensor should only contain the IDs of the answer tokens, with special values (like -100) for padding tokens and the question tokens, so the loss function ignores them.

Let's look at a simplified `VQACollator` from `data/collators.py`.

```python
# From data/collators.py
import torch

class VQACollator(object):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer # Collator uses the tokenizer for text processing
        self.max_length = max_length # Max sequence length for padding/truncation

    def __call__(self, batch):
        # batch is a list of dictionaries, e.g., 
        # [{'image': tensor1, 'text_data': 'Q: A', 'answer': 'B'}, 
        #  {'image': tensor2, 'text_data': 'Q: C D', 'answer': 'E F G'}]
        
        images = [item["image"] for item in batch]
        texts = [item["text_data"] for item in batch] # Questions
        answers = [item["answer"] for item in batch] # Answers

        # Stack images into a single batch tensor (Batch, Channels, Height, Width)
        images = torch.stack(images) 

        # Combine question and answer text for encoding
        # The model is trained to predict the answer given the question
        input_sequences = []
        for i in range(len(batch)):
            input_sequences.append(f"{texts[i]}{answers[i]}")

        # Use the tokenizer to encode the combined sequences into token IDs
        # and pad them to the max_length. Padding is on the LEFT.
        encoded_full_sequences = self.tokenizer.batch_encode_plus(
            input_sequences,
            padding="max_length",
            padding_side="left", # Padding is added at the beginning
            return_tensors="pt", # Return PyTorch tensors
            truncation=True,     # Truncate if sequence is longer than max_length
            max_length=self.max_length,
        )

        input_ids = encoded_full_sequences["input_ids"] # Tensor of token IDs (Batch, max_length)
        attention_mask = encoded_full_sequences["attention_mask"] # Mask (Batch, max_length)

        # Create labels for training. 
        # Labels are the same as input_ids, but shifted by one position
        # because we predict the NEXT token. 
        # We set labels for padding and question tokens to -100 so they are ignored by loss.
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:].clone() # Shift tokens for Causal LM objective
        labels[:, -1] = -100 # Ignore the last token's prediction

        # The VQACollator has complex logic to find where the answer starts
        # after left padding and potentially truncation, and sets labels 
        # before that point to -100. 
        # This ensures loss is ONLY calculated on the answer tokens.
        # ... (Simplified: Logic to set question/padding labels to -100) ...
        # (See full code for details on handling padding and truncation)

        return {
            "image": images,
            "input_ids": input_ids,      # Combined Q+A token IDs (padded)
            "attention_mask": attention_mask, # Mask for Q+A
            "labels": labels             # Target token IDs (A only, shifted, -100 elsewhere)
        }
```

The `__call__` method processes the batch list. It stacks images, uses the `tokenizer` to encode the combined text sequences (`question + answer`), pads them to `max_length`, and critically, generates the `labels` tensor by masking out (setting to -100) the question and padding tokens. This masking is essential because the loss function in `VisionLanguageModel.forward` (as seen in [Chapter 5](05_visionlanguagemodel__vlm__.md)) uses this `-100` value to know which predictions to ignore when calculating how well the model predicted the *answer*.

The `MMStarCollator` is used specifically for the MMStar evaluation dataset and prepares the data in a slightly different format suitable for that benchmark's evaluation script.

## How They Work Together: The Data Pipeline

Here's a simplified flow showing how these components interact to produce a batch ready for the model:

```mermaid
sequenceDiagram
    participant RawData as Raw Data<br>(Image File, Text String)
    participant HFDataset as Hugging Face Dataset Object
    participant nanoVLMDataset as VQADataset / MMStarDataset
    participant Processors as ImageProcessor<br>Tokenizer
    participant ListOfSamples as List of Samples<br>(from DataLoader)
    participant Collator as VQACollator / MMStarCollator
    participant ModelBatch as Batch of Tensors<br>(Ready for Model)

    RawData->>HFDataset: Loaded/Accessed
    HFDataset->>nanoVLMDataset: Passed to __init__
    
    rect rgb(192, 255, 224)
    box DataLoading
        loop For each sample in batch
            DataLoader->>nanoVLMDataset: Call __getitem__(idx)
            nanoVLMDataset->>Processors: Process image
            Processors-->>nanoVLMDataset: Processed Image Tensor
            nanoVLMDataset-->>ListOfSamples: Return sample dict
        end
    end

    rect rgb(255, 238, 204)
    box BatchPreparation
        DataLoader->>Collator: Pass ListOfSamples<br>(Call __call__(batch_list))
        Collator->>Processors: Use Tokenizer to encode & pad text
        Processors-->>Collator: Text Tensors (input_ids, mask)
        Collator->>Collator: Stack image tensors
        Collator->>Collator: Create labels (masking)
        Collator-->>ModelBatch: Return Batch Dictionary of Tensors
    end

    ModelBatch->>VisionLanguageModelClass: Feed to model.forward()

```

1.  You start with raw data sources.
2.  A library like Hugging Face `datasets` loads or provides access to this data.
3.  A nanoVLM `Dataset` class (`VQADataset`, `MMStarDataset`) wraps this data source and knows how to get *one* individual item (`__getitem__`).
4.  Inside `__getitem__`, the `Dataset` uses **Processors** (`image_processor`, and the `tokenizer` conceptually, though encoding happens later) to convert the raw image to a tensor and get the raw text.
5.  A `DataLoader` collects several of these individual processed samples into a list (this list is the `batch` argument to the Collator's `__call__`).
6.  A nanoVLM **Collator** class (`VQACollator`, `MMStarCollator`) takes this list. It uses the **Tokenizer** Processor to encode and pad the text sequences and creates the appropriate `labels` tensor. It also stacks the image tensors.
7.  The Collator returns a single dictionary containing batch tensors (`image`, `input_ids`, `attention_mask`, `labels`), which is finally ready to be fed into the `VisionLanguageModel` for training or inference.

This layered approach keeps the code organized: Datasets handle sample loading, Processors handle raw-to-numerical conversion, and Collators handle batch assembly and label/mask creation.

## Where to Find Them in the Code

You'll see these components used together in `train.py` and `generate.py`.

In `train.py`, the `get_dataloaders` function is responsible for setting up the entire data pipeline:

```python
# Simplified from train.py
def get_dataloaders(train_cfg, vlm_cfg):
    # 1. Get Processors using info from VLMConfig
    tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer)
    image_processor = get_image_processor(vlm_cfg.vit_img_size)

    # 2. Load raw dataset using info from TrainConfig
    train_ds_raw = load_dataset(train_cfg.train_dataset_path, train_cfg.train_dataset_name)
    # ... (split into train/val) ...

    # 3. Create nanoVLM Dataset instances, passing processors
    train_dataset = VQADataset(train_split_raw, tokenizer, image_processor)
    val_dataset = VQADataset(val_split_raw, tokenizer, image_processor)
    test_dataset = MMStarDataset(test_split_raw, tokenizer, image_processor) # Different dataset & class

    # 4. Create Collator instances, passing tokenizer and max_length (from VLMConfig)
    vqa_collator = VQACollator(tokenizer, vlm_cfg.lm_max_length)
    mmstar_collator = MMStarCollator(tokenizer) # Different collator for MMStar

    # 5. Create DataLoaders, passing Dataset, batch_size (from TrainConfig), and Collator
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, collate_fn=vqa_collator, ...)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, collate_fn=vqa_collator, ...)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.mmstar_batch_size, collate_fn=mmstar_collator, ...)

    return train_loader, val_loader, test_loader
```

This function clearly shows the sequence: get processors, load raw data, create Datasets using raw data and processors, create Collators using processors and config, and finally create DataLoaders using Datasets, batch sizes (from `TrainConfig`), and Collators.

In `generate.py`, which is used for inference, you primarily need the **Processors** (`get_tokenizer`, `get_image_processor`) to prepare the single image and prompt you're providing. Batching and complex label creation aren't needed in the same way as training, though the model internally handles sequences and masks during generation.

```python
# Simplified from generate.py main()
def main():
    # ... load model (which includes its config: model.cfg) ...

    # Get Processors using info from the loaded model's VLMConfig
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer)
    image_processor = get_image_processor(model.cfg.vit_img_size)

    # Prepare the input image using the image processor
    pil_image = Image.open(args.image).convert("RGB")
    image_tensor = image_processor(pil_image).unsqueeze(0).to(device) # Process and add batch dim

    # Prepare the input text (prompt) using the tokenizer
    template = f"Question: {args.prompt} Answer:"
    encoded = tokenizer.batch_encode_plus([template], return_tensors="pt")
    input_ids = encoded["input_ids"].to(device) # Token IDs
    attention_mask = encoded["attention_mask"].to(device) # Mask

    # Now feed image_tensor, input_ids, and attention_mask to model.generate(...)
    # ...
```

Here, the processors are used directly to prepare the inputs *before* calling `model.generate`.

## Conclusion

In this chapter, we've demystified the data handling pipeline in nanoVLM. We learned that **Datasets** are responsible for loading individual samples, **Processors** convert raw images and text into the numerical formats the model needs, and **Collators** group these processed samples into efficient batches, handling crucial steps like padding and creating training labels by masking tokens. These components, orchestrated by the `DataLoader` and configured by `TrainConfig` and `VLMConfig`, form the essential link between the raw data and the model's training and inference steps.

With this chapter, we've covered the core components of the nanoVLM project: the configuration files (`VLMConfig`, `TrainConfig`), the model architecture modules (`ViT`, `MP`, `LM`, `VisionLanguageModel`), and the data preparation pipeline. You now have a solid understanding of the main pieces that make nanoVLM work!

