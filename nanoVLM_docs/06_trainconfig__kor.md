# 챕터 6: TrainConfig

nanoVLM 튜토리얼에 다시 오신 것을 환영합니다! 지금까지 우리는 [VisionLanguageModel (VLM)](05_vision_language_model__vlm__.md)에 대한 이해를 쌓아왔습니다: [VLMConfig](01_vlmconfig_.md)에서 그 기초 청사진을 탐구하고, 핵심 구성 요소들([Vision Transformer (ViT)](02_vision_transformer__vit__.md), [Modality Projector (MP)](03_modality_projector__mp__.md), [Language Model (LM)](04_language_model__lm__.md))을 만났으며, 메인 [VisionLanguageModel](05_vision_language_model__vlm__.md) 클래스가 이들을 어떻게 하나로 모으는지 보았습니다.

이제 완전한 모델이 구축되어 준비되었습니다. 다음은 무엇일까요? 모델을 **학습**시켜야 합니다!

## "조리 지침": TrainConfig란 무엇인가?

방금 매우 복잡한 새로운 기계([VisionLanguageModel](05_vision_language_model__vlm__.md))를 조립했다고 상상해보세요. 이 기계는 청사진([VLMConfig](01_vlmconfig_.md))에 따라 구축되었고 모든 정교한 부품들([ViT](02_vision_transformer__vit__.md), [MP](03_modality_projector__mp__.md), [LM](04_language_model__lm__.md))을 가지고 있습니다. 하지만 지금은 Vision-Language 작업을 수행하는 방법을 *알지 못합니다*. 데이터를 사용하여 가르쳐야 합니다.

학습은 모델에 많은 예제(이미지-텍스트 쌍)를 제공하고 내부 설정(가중치)을 조정하여 주어진 이미지와 프롬프트에 대해 올바른 텍스트 출력을 생성하는 능력을 향상시키는 과정입니다.

이 학습 과정 자체도 *학습이 어떻게 이루어져야 하는지*에 대한 지침, 즉 계획이 필요합니다. 이 계획은 **`TrainConfig`**에 저장됩니다.

`TrainConfig`를 VLM 학습을 위한 **"조리 지침"**이라고 생각하세요. 이는 학습 루프를 제어하는 모든 특정 설정(일반적으로 **하이퍼파라미터**라고 불림)을 담고 있습니다. 이러한 설정들은 학습 과정에 다음과 같은 것들을 알려줍니다:

*   모델이 얼마나 빠르게 학습해야 하는가? (Learning Rate)
*   한 번에 몇 개의 예제를 보아야 하는가? (Batch Size)
*   전체 데이터셋을 몇 번 통과해야 하는가? (Epochs)
*   학습 데이터는 어디에서 찾을 수 있는가? (Dataset Paths)
*   진행 보고서를 저장해야 하는가? (Logging Options)

[VLMConfig](01_vlmconfig_.md)를 사용하여 모델 자체를 생성하는 것처럼, `TrainConfig`를 사용하여 그 모델이 *어떻게* 학습될지 정의합니다. 이는 nanoVLM의 두 번째 중요한 설정 파일입니다.

## TrainConfig를 사용하여 모델 학습하기

`TrainConfig`가 사용되는 주요 장소는 `train.py` 스크립트입니다. 이 스크립트는 전체 학습 과정을 조율하는 메인 `train` 함수를 포함합니다.

먼저 `TrainConfig`를 임포트해야 합니다:

```python
# From train.py or models/config.py
from models.config import TrainConfig
```

그런 다음 `TrainConfig`의 인스턴스를 생성합니다. `VLMConfig`처럼, 이는 합리적인 기본값들을 가지고 있습니다.

```python
# 기본 설정으로 TrainConfig 인스턴스 생성
train_cfg = TrainConfig()

print(train_cfg)
```

`train_cfg`를 출력하면 `batch_size=256`, `epochs=5`, `lr_mp=2e-3` 등과 같은 모든 기본 학습 하이퍼파라미터를 볼 수 있습니다.

종종 사용 가능한 하드웨어(예: GPU 메모리), 사용 중인 특정 데이터셋, 또는 실험 목표에 따라 이러한 기본값을 조정하고 싶을 것입니다. 인스턴스 생성 시 기본값을 재정의할 수 있습니다:

```python
# 기본 배치 크기와 에폭을 재정의하여 TrainConfig 인스턴스 생성
custom_train_cfg = TrainConfig(batch_size=128, epochs=10)

print(custom_train_cfg.batch_size) # 출력: 128
print(custom_train_cfg.epochs)    # 출력: 10
```

`train.py`에서 `main` 함수는 명령줄 인수를 파싱하고 이를 사용하여 `VLMConfig`와 `TrainConfig` 모두의 기본 설정을 잠재적으로 재정의한 후 핵심 `train` 함수에 전달합니다.

```python
# train.py main()에서 단순화됨
def main():
    parser = argparse.ArgumentParser()
    # --lr_mp, --batch_size 등의 인수 추가
    parser.add_argument('--lr_mp', type=float, help='Learning rate for MP')
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    # ... 다른 인수들 ...
    args = parser.parse_args()

    # 기본 설정 생성
    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig() # 기본 TrainConfig

    # 명령줄 인수를 기반으로 train_cfg 설정 재정의
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    # ... 다른 설정들 재정의 ...

    print("--- Train Config ---")
    print(train_cfg) # 사용되는 최종 설정 출력

    # 메인 학습 함수에 BOTH 설정 전달
    train(train_cfg, vlm_cfg)
```

`train` 함수는 이 `train_cfg` 객체(와 `vlm_cfg`)를 받아 학습 루프 전체에서 그 값들을 사용합니다.

```python
# train.py train() 함수 시그니처에서 단순화됨
def train(train_cfg: TrainConfig, vlm_cfg: VLMConfig):
    # train_cfg는 이제 이 학습 실행을 위한 모든 설정을 보유
    # ... 학습 설정과 루프의 나머지 부분 ...
    pass
```

이 패턴은 학습 로직을 깔끔하게 유지합니다. 학습을 위한 모든 손잡이와 다이얼이 `train_cfg` 객체 내에 깔끔하게 정리되어 있기 때문입니다.

## TrainConfig의 주요 설정들

`TrainConfig`에 정의된 중요한 하이퍼파라미터들과 그들이 제어하는 것들을 살펴보겠습니다:

*   **`lr_mp`와 `lr_backbones`:** 학습률. 이는 아마도 가장 중요한 설정들일 것입니다. 이들은 계산된 손실을 기반으로 모델의 가중치를 조정할 때 옵티마이저가 취하는 단계의 크기를 결정합니다. 더 높은 학습률은 더 빠르지만 잠재적으로 불안정한 학습을 의미하고, 더 낮은 학습률은 더 느리지만 잠재적으로 더 안정적인 학습을 의미합니다. nanoVLM이 *두 가지* 다른 학습률을 사용한다는 점에 주목하세요: 하나는 [Modality Projector (MP)](03_modality_projector__mp__.md)를 위한 것(`lr_mp`)이고, 다른 하나는 [ViT](02_vision_transformer__vit__.md)와 [LM](04_language_model__lm__.md) 백본을 위한 것(`lr_backbones`)입니다. 이는 MP가 보통 무작위로 초기화되고 상대적으로 빠르게 학습해야 하는 반면, 백본들은 종종 사전 학습되어 있고 미세 조정만 필요하기 때문입니다.
*   **`batch_size`:** 한 번의 순전파와 역전파에서 함께 처리되는 이미지-텍스트 쌍의 수입니다. 더 큰 배치 크기는 더 안정적인 그래디언트 추정으로 이어질 수 있지만 훨씬 더 많은 GPU 메모리가 필요합니다.
*   **`epochs`:** 하나의 에폭은 학습 과정이 *전체* 학습 데이터셋을 한 번 통과했다는 것을 의미합니다. `epochs`는 이 반복이 발생하는 총 횟수를 설정합니다. 더 많은 에폭은 종종 더 나은 성능으로 이어지지만 더 오래 걸립니다.
*   **`train_dataset_path`, `train_dataset_name`, `test_dataset_path`:** 이 문자열들은 데이터가 위치한 곳을 지정합니다(예: 디렉토리 경로나 Hugging Face Hub의 데이터셋 이름). `train_dataset_name`은 여러 데이터셋을 결합하여 학습할 수 있기 때문에 튜플입니다(코드 스니펫에서 볼 수 있듯이).
*   **`val_ratio`:** 검증을 위해 따로 둘 학습 데이터의 비율(모델이 엄격하게 학습하지 않은 데이터에 대한 학습 중 성능 확인).
*   **`eval_in_epochs`:** 평가(MMStar와 같은 별도의 테스트 세트에서 모델 실행)가 학습 에폭 *동안* 주기적으로 발생할지, 아니면 마지막에만 발생할지를 제어하는 불리언 플래그입니다.
*   **`compile`:** `torch.compile`을 활성화하는 불리언 플래그로, 모델의 계산 그래프를 최적화하여 학습 속도를 크게 향상시킬 수 있습니다.
*   **`resume_from_vlm_checkpoint`:** 학습이 개별 백본 가중치를 로드하는 대신 이전에 저장된 전체 VLM 체크포인트(`VLMConfig`의 `vlm_checkpoint_path`로 지정됨)에서 가중치를 로드하여 시작해야 하는지 여부를 나타내는 불리언 플래그입니다.
*   **`log_wandb`와 `wandb_entity`:** 학습 메트릭과 결과를 Weights & Biases (WandB)라는 인기 있는 실험 추적 플랫폼에 로깅하는 것과 관련된 설정입니다.

이들은 주요 설정들 중 일부일 뿐입니다. 이러한 값들을 조정하는 것이 학습 실험을 제어하고 VLM을 가르치는 최선의 방법을 찾는 방법입니다.

## 내부 작동: TrainConfig가 학습을 지시하는 방법

`VLMConfig`처럼, `TrainConfig`는 `dataclass`입니다. 이는 단순히 설정 값들을 보유합니다. 이러한 값들을 *사용하는* 실제 로직은 학습 스크립트 내부, 주로 `train.py`의 `train` 함수 내에 있습니다.

`TrainConfig`가 학습 과정을 통해 어떻게 흐르는지 보여주는 단순화된 시퀀스 다이어그램입니다:

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

`train_cfg` 객체는 시작 시 한 번 생성된 다음 학습 설정의 다른 부분들로 전달됩니다:

1.  **데이터 로딩:** `get_dataloaders` 함수는 `train_cfg.train_dataset_path`, `train_cfg.train_dataset_name`, `train_cfg.val_ratio`, `train_cfg.batch_size`, `train_cfg.mmstar_batch_size`를 읽어 데이터가 어떻게 로드되고 배치되는지 구성합니다.
2.  **옵티마이저:** `optim.AdamW` 옵티마이저는 서로 다른 파라미터 그룹(MP와 백본용)으로 초기화되며, 각각의 학습률에 `train_cfg.lr_mp`와 `train_cfg.lr_backbones`를 사용합니다.
3.  **학습 루프:** 메인 루프는 `train_cfg.epochs` 횟수만큼 반복됩니다. 루프 내부에서 배치는 데이터로더에서 구성된 `batch_size`(이는 `train_cfg`에서 얻음)를 사용하여 처리됩니다. 평가 빈도는 `train_cfg.eval_in_epochs`에 의해 제어됩니다.
4.  **로깅:** `train_cfg.log_wandb`가 true이면, 스크립트는 `train_cfg.wandb_entity`를 사용하여 WandB 로깅을 초기화하고 학습 진행 상황에 따라 손실과 정확도와 같은 메트릭을 WandB로 전송합니다.

`train.py`에서 이 사용을 보여주는 몇 가지 단순화된 코드 스니펫을 살펴보겠습니다.

### 데이터 로더 가져오기

```python
# train.py get_dataloaders(...)에서 단순화됨
def get_dataloaders(train_cfg, vlm_cfg):
    # ... tokenizer와 image_processor 설정 ...

    # train_cfg에서 경로/이름을 사용하여 데이터셋 로드
    train_ds = load_dataset(train_cfg.train_dataset_path, train_cfg.train_dataset_name)
    # ... train_cfg.val_ratio를 사용하여 연결, 섞기, 분할 ...

    # train_cfg에서 batch_size를 사용하여 DataLoader 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size, # 여기서 사용됨
        shuffle=True,
        # ... 다른 DataLoader 인수들 ...
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size, # 여기서 사용됨
        shuffle=False,
        # ... 다른 DataLoader 인수들 ...
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_cfg.mmstar_batch_size, # 테스트 세트용으로 여기서 사용됨
        shuffle=False,
        # ... 다른 DataLoader 인수들 ...
    )
    return train_loader, val_loader, test_loader
```

`batch_size`, 데이터셋 경로, 검증 분할 비율은 모두 데이터 파이프라인을 설정하기 위해 `train_cfg` 객체에서 직접 읽습니다.

### 옵티마이저 설정

```python
# train.py train() 함수에서 단순화됨
def train(train_cfg, vlm_cfg):
    # ... 데이터로더 가져오기, 모델 초기화 ...

    # train_cfg에서 서로 다른 학습률로 파라미터 그룹 정의
    param_groups = [{'params': model.MP.parameters(), 'lr': train_cfg.lr_mp}, # MP용 LR
                    {'params': list(model.decoder.parameters()) + list(model.vision_encoder.parameters()), 'lr': train_cfg.lr_backbones}] # 백본용 LR
    
    optimizer = optim.AdamW(param_groups)

    # ... 학습 루프의 나머지 부분 ...
```

여기서 `train_cfg.lr_mp`와 `train_cfg.lr_backbones`는 옵티마이저에게 Modality Projector와 시각 및 언어 백본의 가중치를 얼마나 적극적으로 업데이트할지 알려주는 데 사용됩니다.

### 학습 루프

```python
# train.py train() 함수에서 단순화됨
def train(train_cfg, vlm_cfg):
    # ... 설정 코드 ...

    # 지정된 에폭 수만큼 루프
    for epoch in range(train_cfg.epochs): # 여기서 train_cfg.epochs 사용
        model.train()
        total_train_loss = 0

        # 배치 반복 (크기는 DataLoader에서 train_cfg.batch_size에 의해 결정됨)
        for batch in train_loader:
            # ... 데이터를 디바이스로 ...
            
            optimizer.zero_grad()

            # 순전파 (VisionLanguageModel은 내부적으로 VLMConfig 사용)
            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

            loss.backward()
            
            # 단계에 따라 학습률 조정 (train_cfg.lr_mp, train_cfg.lr_backbones 사용)
            # ... get_lr(global_step, train_cfg.lr_mp, total_steps) ...
            optimizer.param_groups[0]['lr'] = adj_lr_mp
            optimizer.param_groups[1]['lr'] = adj_lr_backbones

            optimizer.step()

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # 학습 중 주기적 평가
            if train_cfg.eval_in_epochs and global_step % 250 == 0: # 여기서 train_cfg.eval_in_epochs 사용
                # ... 평가 함수 호출 ...
                pass
            
            # 로깅
            if train_cfg.log_wandb: # 여기서 train_cfg.log_wandb 사용
                # ... 메트릭 로깅 ...
                pass

            global_step += 1

        # ... 에폭 종료 계산 및 로깅 ...
```

메인 학습 루프 구조(에폭 수, 언제 평가할지, 언제 로깅할지)는 `train_cfg`에서 읽은 값에 의해 직접 제어됩니다.

## 내부 작동: `TrainConfig` 정의

마지막으로, `models/config.py`에서 `TrainConfig` `dataclass`의 정의를 살펴보겠습니다.

```python
# From models/config.py
from dataclasses import dataclass

@dataclass # 이 데코레이터가 dataclass로 만듦
class TrainConfig:
    # 타입과 기본값이 있는 학습 하이퍼파라미터
    lr_mp: float = 2e-3           # Modality Projector의 학습률
    lr_backbones: float = 1e-4    # ViT와 LM 백본의 학습률
    data_cutoff_idx: int = None   # 데이터셋의 처음 N개 샘플만 사용
    val_ratio: float = 0.01       # 검증을 위한 데이터 비율
    batch_size: int = 256         # 메인 학습을 위한 배치 크기
    mmstar_batch_size: int = 32   # MMStar 평가를 위한 특정 배치 크기
    eval_in_epochs: bool = True   # 학습 중 평가를 실행할지 여부
    epochs: int = 5               # 총 학습 에폭 수
    compile: bool = True          # torch.compile 활성화
    resume_from_vlm_checkpoint: bool = False # 저장된 VLM 체크포인트에서 시작
    train_dataset_path: str = 'HuggingFaceM4/the_cauldron' # 학습 데이터를 위한 HF Hub 경로
    train_dataset_name: tuple[str, ...] = ("ai2d", "aokvqa", "chart2text", ...) # 사용할 특정 데이터셋 이름들
    test_dataset_path: str = "Lin-Chen/MMStar" # 테스트 데이터를 위한 HF Hub 경로
    wandb_entity: str = "HuggingFace" # WandB 엔티티 이름
    log_wandb: bool = True        # WandB 로깅 활성화
```

이 정의는 단순히 모든 구성 가능한 학습 파라미터들을 그들의 데이터 타입과 기본값과 함께 나열합니다. `train.py` 스크립트는 이 클래스를 임포트하고 이의 인스턴스를 사용하여 이러한 값들을 가져와 학습 과정을 주도합니다.

## 결론

이 챕터에서 우리는 [VisionLanguageModel (VLM)](05_vision_language_model__vlm__.md)을 학습시키기 위한 "조리 지침"을 제공하는 필수적인 설정 객체인 `TrainConfig`를 소개했습니다. 우리는 이것이 학습률, 배치 크기, 에폭, 데이터셋 경로와 같은 중요한 하이퍼파라미터를 보유하는 방법을 보았습니다. 우리는 `train.py`가 `TrainConfig` 인스턴스를 사용하여 데이터 로더를 설정하고, 옵티마이저를 구성하고, 학습 루프를 제어하고, 로깅을 관리하는 방법을 배웠습니다. `TrainConfig`를 이해하는 것은 nanoVLM에서 학습 과정을 실행하거나 커스터마이즈할 수 있는 핵심입니다.

`TrainConfig`는 *어떤* 데이터셋을 사용할지와 *어떻게* 배치할지 알려주지만, 데이터가 어떻게 로드되고, 처리되고, 모델을 위해 준비되는지의 세부 사항은 다루지 않습니다. 그것이 우리의 다음 챕터의 주제입니다.

[다음 챕터: 데이터 처리 (데이터셋, 콜레이터, 프로세서)](07_data_handling__datasets__collators__processors__.md) 