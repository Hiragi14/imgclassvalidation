# tests/test_fvcore_flops.py

import torch
import torch.nn as nn

from imgclassvalidation.extras import fvcore_flops


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


def test_fvcore_flops_returns_total_flops():
    model = SimpleCNN()
    example_input = torch.randn(1, 3, 32, 32)
    device = torch.device("cpu")

    result = fvcore_flops(model, example_input, device)

    assert "fvcore_flops_total" in result
    assert "fvcore_error" in result

    assert result["fvcore_error"] is None
    assert isinstance(result["fvcore_flops_total"], float)
    assert result["fvcore_flops_total"] > 0.0


def test_fvcore_flops_sets_model_to_eval():
    model = SimpleCNN()
    model.train()

    example_input = torch.randn(1, 3, 32, 32)
    device = torch.device("cpu")

    _ = fvcore_flops(model, example_input, device)

    assert model.training is False


def test_fvcore_flops_handles_forward_error():
    class BrokenModel(nn.Module):
        def forward(self, x):
            raise RuntimeError("intentional error")

    model = BrokenModel()
    example_input = torch.randn(1, 3, 32, 32)
    device = torch.device("cpu")

    result = fvcore_flops(model, example_input, device)

    assert result["fvcore_flops_total"] is None
    assert result["fvcore_error"] is not None


def test_fvcore_flops_moves_input_to_device():
    model = SimpleCNN()
    example_input = torch.randn(1, 3, 32, 32)

    device = torch.device("cpu")

    result = fvcore_flops(model, example_input, device)

    assert result["fvcore_error"] is None
    assert result["fvcore_flops_total"] is not None
