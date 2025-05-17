# 챕터 7: 데이터 처리 (데이터셋, 콜레이터, 프로세서)

nanoVLM 튜토리얼에 다시 오신 것을 환영합니다! 이전 챕터인 [챕터 6: TrainConfig](06_trainconfig_.md)에서 우리는 [VisionLanguageModel (VLM)](05_visionlanguagemodel__vlm__.md)을 *어떻게* 학습시킬지 알려주는 설정(`TrainConfig`)에 대해 배웠습니다 - 학습률, 배치 크기, 사용할 데이터셋 등에 관한 내용이었죠.

하지만 실제로 파일이나 온라인 저장소에서 데이터(이미지와 텍스트)를 *가져오고*, 모델이 이해할 수 있도록 준비하고(이미지를 텐서로 변환하고, 텍스트를 토큰 ID로 변환), 학습을 위해 효율적으로 그룹화하는 것은 어떻게 할까요?

이것이 데이터 처리 구성 요소들의 역할입니다: **데이터셋**, **프로세서**, 그리고 **콜레이터**.

## 모델을 위한 재료 준비하기

복잡한 기계(우리의 VLM)를 만드는 공장을 운영한다고 상상해보세요. 각 기계는 학습하고 작동하기 위해 전처리된 부품들(데이터 배치)의 지속적인 공급이 필요합니다. 원자재(JPEG 이미지 파일이나 일반 텍스트 문자열과 같은)를 기계에 그냥 던져넣을 수는 없습니다. 먼저 준비가 필요합니다!

이 준비 과정에는 여러 단계가 있습니다:

1.  **개별 부품 수집:** 각 원자재(특정 이미지 파일과 그에 해당하는 텍스트 캡션이나 질문과 같은)를 어디서 찾을지 알아야 합니다. 이것은 **데이터셋**이 처리합니다.
2.  **부품 형상화:** 원자재는 기계가 처리할 수 있는 표준 형식으로 형상화되고 측정되어야 합니다(이미지를 고정된 크기로 조정하거나 텍스트를 숫자 ID로 변환하는 것과 같은). 이것은 **프로세서**가 처리합니다.
3.  **그룹화와 배열:** 여러 개의 준비된 부품들을 모아서 기계가 배치로 효율적으로 처리할 수 있도록 깔끔하게 배열해야 합니다. 여기에는 패딩을 추가하여 배치의 모든 텍스트 시퀀스가 같은 길이를 가지도록 하는 것이 포함됩니다. 이것은 **콜레이터**가 처리합니다.

데이터셋, 프로세서, 콜레이터는 nanoVLM에서 함께 작동하여 [VisionLanguageModel](05_vision_language_model__vlm__.md)이 학습과 추론에 필요한 완벽하게 포맷된 데이터 배치를 만듭니다.

## 주요 구성 요소

이 세 가지 유형의 구성 요소들의 역할을 자세히 살펴보겠습니다:

*   **데이터셋:** 소스(파일이나 라이브러리의 데이터셋 객체와 같은)에서 *개별* 데이터 샘플(하나의 이미지, 하나의 텍스트 쌍)을 로드하는 역할을 합니다. 이들은 단일 예제가 어떻게 생겼는지 정의합니다.
*   **프로세서:** 원시 데이터(PIL Image 객체나 텍스트 문자열과 같은)를 신경망이 이해하는 수치 형식(PyTorch 픽셀 텐서나 토큰 ID와 같은)으로 변환하는 역할을 합니다.
*   **콜레이터:** 이러한 개별 처리된 샘플들의 *리스트*를 가져와서 단일 *배치* 텐서로 결합하는 역할을 합니다. 여기서 시퀀스를 균일한 길이로 패딩하고 학습을 위한 레이블을 생성하는 것과 같은 작업이 이루어집니다.

이제 이것이 nanoVLM 코드에서 어떻게 작동하는지 살펴보겠습니다.

## 데이터셋: 개별 샘플 로드하기

nanoVLM에서 데이터셋 클래스는 PyTorch의 `torch.utils.data.Dataset`을 상속받습니다. 이들의 주요 작업은 두 가지 메서드를 구현하는 것입니다:

*   `__len__`: 데이터셋의 총 샘플 수를 반환합니다.
*   `__getitem__`: 인덱스(숫자)를 받아 하나의 단일 샘플(이미지, 관련 텍스트 등)을 반환합니다.

nanoVLM은 서로 다른 데이터 형식이나 작업에 대해 서로 다른 데이터셋 클래스를 사용합니다. 예를 들어, 일반적인 Visual Question Answering 데이터(Hugging Face의 `the_cauldron`의 데이터셋과 같은)를 위한 `VQADataset`과 MMStar 벤치마크에서 사용되는 특정 형식을 위한 `MMStarDataset`이 있습니다.

`data/datasets.py`에서 `VQADataset`의 단순화된 예제를 살펴보겠습니다.

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

`__getitem__` 메서드는 인덱스(`idx`)를 받아 기본 데이터셋에서 원시 항목을 검색하고, `image_processor`(다음에 볼 것입니다)를 사용하여 원시 이미지를 텐서로 변환하고, 텍스트를 포맷팅하여 처리된 이미지 텐서와 원시 텍스트 문자열(질문과 답변)을 포함하는 딕셔너리를 반환합니다.

`MMStarDataset`은 매우 유사하지만 MMStar 평가 데이터셋의 특정 형식에 맞게 조정되어 있습니다. 이것도 입력을 준비하기 위해 `image_processor`와 `tokenizer`를 사용합니다.

## 프로세서: 원시 데이터 변환하기

프로세서는 원시 입력(`PIL.Image` 객체나 Python 문자열과 같은)을 모델이 처리할 수 있는 수치 텐서로 변환하는 도구입니다.

nanoVLM에서 `data/processors.py` 파일은 필요한 프로세서를 가져오는 간단한 함수들을 포함합니다:

*   `get_image_processor`: 이미지의 크기를 조정하고 PyTorch 텐서로 변환하는 함수(특히 `torchvision.transforms.Compose` 객체)를 반환합니다.
*   `get_tokenizer`: 텍스트 문자열을 수치 토큰 ID 시퀀스로 변환하고 그 반대도 가능하게 하는 Hugging Face `transformers` 토크나이저 객체를 반환합니다.

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

이러한 프로세서들은 모든 이미지가 같은 크기를 가지고 모든 텍스트가 [Vision Transformer (ViT)](02_vision_transformer__vit__.md)와 [Language Model (LM)](04_language_model__lm__.md)이 기대하는 입력과 일치하는 숫자로 표현되도록 보장하기 때문에 중요합니다. `VQADataset` 스니펫에서 볼 수 있듯이, `image_processor`는 데이터셋의 `__getitem__` *내부에서* 사용됩니다. 그러나 `tokenizer`는 주로 **콜레이터**가 텍스트 배치를 처리하는 데 사용되며, 추론 스크립트([generate.py](generate.py))에서도 프롬프트와 출력을 인코딩/디코딩하는 데 사용됩니다.

## 콜레이터: 배치 처리와 패딩

콜레이터는 배치 처리의 마법이 일어나는 곳입니다. `torch.utils.data.DataLoader`는 데이터셋과 `batch_size`를 받아 `Dataset.__getitem__`을 여러 번 호출하여 개별 샘플들의 리스트를 얻은 다음, 이 리스트를 콜레이터의 `__call__` 메서드에 전달합니다. 콜레이터의 작업은 이 딕셔너리 리스트(`VQADataset.__getitem__`이 반환하는 것과 같은)를 가져와서 각 텐서에 배치 차원이 추가된 PyTorch 텐서들의 *단일* 딕셔너리로 결합하는 것입니다.

이것은 이미지의 경우 상대적으로 간단합니다(`image_processor` 덕분에 모두 같은 크기이므로 그냥 쌓으면 됩니다). 그러나 텍스트 시퀀스(질문과 답변)는 거의 확실히 다른 길이를 가질 것입니다. 신경망은 균일한 모양의 텐서를 필요로 하므로, 콜레이터는 **패딩**을 처리해야 합니다 - 배치에서 가장 긴 시퀀스와 같은 길이가 되도록 짧은 시퀀스에 특별한 "패딩" 토큰을 추가하는 것입니다. 또한 모델이 어떤 토큰이 실제이고 어떤 것이 단지 패딩인지 알 수 있도록 **어텐션 마스크**를 생성해야 합니다.

또한 학습을 위해 콜레이터는 **레이블**을 준비해야 합니다. nanoVLM의 학습 설정에서, 우리는 모델이 이미지와 질문이 주어졌을 때 *답변* 토큰을 예측하도록 학습시킵니다. 이는 레이블 텐서가 답변 토큰의 ID만 포함해야 하며, 손실 함수가 이를 무시하도록 패딩 토큰과 질문 토큰에 대해 특별한 값(예: -100)을 가져야 한다는 것을 의미합니다.

`data/collators.py`에서 단순화된 `VQACollator`를 살펴보겠습니다.

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

`__call__` 메서드는 배치 리스트를 처리합니다. 이미지를 쌓고, `tokenizer`를 사용하여 결합된 텍스트 시퀀스(`질문 + 답변`)를 인코딩하고, `max_length`로 패딩하며, 중요한 것은 질문과 패딩 토큰을 마스킹(값을 -100으로 설정)하여 `labels` 텐서를 생성합니다. 이 마스킹은 [챕터 5](05_visionlanguagemodel__vlm__.md)에서 본 `VisionLanguageModel.forward`의 손실 함수가 모델이 *답변*을 얼마나 잘 예측했는지 계산할 때 어떤 예측을 무시해야 하는지 알기 위해 이 `-100` 값을 사용하기 때문에 필수적입니다.

`MMStarCollator`는 MMStar 평가 데이터셋을 위해 특별히 사용되며, 해당 벤치마크의 평가 스크립트에 적합한 약간 다른 형식으로 데이터를 준비합니다.

## 함께 작동하는 방법: 데이터 파이프라인

이 구성 요소들이 모델을 위한 배치를 생성하기 위해 어떻게 상호작용하는지 보여주는 단순화된 흐름입니다:

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

1.  원시 데이터 소스로 시작합니다.
2.  Hugging Face `datasets`와 같은 라이브러리가 이 데이터를 로드하거나 접근을 제공합니다.
3.  nanoVLM `Dataset` 클래스(`VQADataset`, `MMStarDataset`)가 이 데이터 소스를 감싸고 *하나의* 개별 항목을 가져오는 방법(`__getitem__`)을 알고 있습니다.
4.  `__getitem__` 내부에서 `Dataset`은 **프로세서**(`image_processor`, 그리고 개념적으로는 `tokenizer`도, 비록 인코딩은 나중에 일어나지만)를 사용하여 원시 이미지를 텐서로 변환하고 원시 텍스트를 가져옵니다.
5.  `DataLoader`는 이러한 개별 처리된 샘플들을 리스트로 수집합니다(이 리스트가 콜레이터의 `__call__`에 전달되는 `batch` 인자입니다).
6.  nanoVLM **콜레이터** 클래스(`VQACollator`, `MMStarCollator`)가 이 리스트를 가져옵니다. 이것은 **토크나이저** 프로세서를 사용하여 텍스트 시퀀스를 인코딩하고 패딩하며 적절한 `labels` 텐서를 생성합니다. 또한 이미지 텐서들을 쌓습니다.
7.  콜레이터는 배치 텐서들(`image`, `input_ids`, `attention_mask`, `labels`)을 포함하는 단일 딕셔너리를 반환하며, 이것이 마침내 학습이나 추론을 위해 `VisionLanguageModel`에 공급될 준비가 됩니다.

이 계층적 접근 방식은 코드를 체계적으로 유지합니다: 데이터셋은 샘플 로딩을 처리하고, 프로세서는 원시-수치 변환을 처리하며, 콜레이터는 배치 조립과 레이블/마스크 생성을 처리합니다.

## 코드에서 찾을 수 있는 위치

이러한 구성 요소들이 함께 사용되는 것을 `train.py`와 `generate.py`에서 볼 수 있습니다.

`train.py`에서 `get_dataloaders` 함수는 전체 데이터 파이프라인을 설정하는 역할을 합니다:

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

이 함수는 프로세서 가져오기, 원시 데이터 로드하기, 원시 데이터와 프로세서를 사용하여 데이터셋 생성하기, 프로세서와 설정을 사용하여 콜레이터 생성하기, 그리고 마지막으로 데이터셋, 배치 크기(`TrainConfig`에서), 콜레이터를 사용하여 데이터로더 생성하기의 순서를 명확하게 보여줍니다.

추론에 사용되는 `generate.py`에서는 주로 **프로세서**(`get_tokenizer`, `get_image_processor`)가 필요합니다. 이것들은 제공하는 단일 이미지와 프롬프트를 준비하는 데 사용됩니다. 학습과 같은 방식으로 배치 처리와 복잡한 레이블 생성이 필요하지는 않지만, 모델은 생성 중에 내부적으로 시퀀스와 마스크를 처리합니다.

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

여기서 프로세서는 `model.generate`를 호출하기 *전에* 입력을 준비하는 데 직접 사용됩니다.

## 결론

이 챕터에서 우리는 nanoVLM의 데이터 처리 파이프라인을 명확하게 설명했습니다. 우리는 **데이터셋**이 개별 샘플을 로드하는 역할을 하고, **프로세서**가 원시 이미지와 텍스트를 모델이 필요로 하는 수치 형식으로 변환하며, **콜레이터**가 이러한 처리된 샘플들을 효율적인 배치로 그룹화하고 패딩과 토큰 마스킹을 통한 학습 레이블 생성과 같은 중요한 단계를 처리한다는 것을 배웠습니다. 이러한 구성 요소들은 `DataLoader`에 의해 조율되고 `TrainConfig`와 `VLMConfig`에 의해 구성되어 원시 데이터와 모델의 학습 및 추론 단계 사이의 필수적인 연결을 형성합니다.

이 챕터로 우리는 nanoVLM 프로젝트의 핵심 구성 요소들을 모두 다루었습니다: 설정 파일들(`VLMConfig`, `TrainConfig`), 모델 아키텍처 모듈들(`ViT`, `MP`, `LM`, `VisionLanguageModel`), 그리고 데이터 준비 파이프라인. 이제 nanoVLM이 작동하게 만드는 주요 부분들에 대한 견고한 이해를 가지게 되었습니다! 