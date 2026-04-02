import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from ImgClassValidation.types import EvalConfig
from ImgClassValidation.engines import evaluate_classification  # ← あなたの関数の配置に合わせて変更


def main():
    # -------------------------
    # 1) Device / Config
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = EvalConfig(
        device=device,
        amp=(device.type == "cuda"),  # CUDAならAMP on（任意）
        non_blocking=True,
        topk=(1, 5),
        criterion=nn.CrossEntropyLoss(),
    )

    # -------------------------
    # 2) Dataset / DataLoader (Evaluation用)
    # -------------------------
    # ResNetは ImageNet 用に 224x224 前提なので、CIFAR10(32x32)を224に上げる簡易設定
    # ※「正しい精度」を目指すものではなく、APIテスト用です
    tfm = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    test_ds = datasets.CIFAR10(root="/DeepLearning/Dataset/torchvision/CIFAR10", train=False, download=True, transform=tfm)
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,           # 評価なので基本False
        num_workers=8,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True, # 大規模評価寄り（環境によりFalseでも可）
    )

    # -------------------------
    # 3) Model (Torchvision)
    # -------------------------
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR10 -> 10 classes
    model.to(device)

    # -------------------------
    # 4) Run evaluation
    # -------------------------
    result = evaluate_classification(
        model=model,
        dataloader=test_loader,
        config=config,
        # output_transform=None,  # 通常は不要（logits, yがそのまま）
    )

    # -------------------------
    # 5) Print report
    # -------------------------
    print("\n=== Metrics ===")
    for k, v in result.metrics.items():
        print(f"{k}: {v:.6f}")

    print("\n=== Extras ===")
    for k, v in result.extras.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
