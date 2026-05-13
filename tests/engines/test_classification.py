# tests/test_evaluator.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from imgclassvalidation.engines.classification import create_classification_evaluator
from imgclassvalidation.types import EvalConfig


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


def make_dataloader():
    x = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    y = torch.tensor([0, 1, 2, 1])

    dataset = TensorDataset(x, y)

    return DataLoader(dataset, batch_size=2, shuffle=False)


def test_create_classification_evaluator_runs(monkeypatch):
    import imgclassvalidation.engines.classification as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "fvcore_flops",
        lambda model, example_input, device: {
            "fvcore_flops_total": 123.0,
            "fvcore_error": None,
        },
    )

    model = SimpleClassifier()
    dataloader = make_dataloader()

    config = EvalConfig(
        device=torch.device("cpu"),
        amp=False,
        non_blocking=False,
        topk=(1, 2),
        criterion=nn.CrossEntropyLoss(),
    )

    evaluator = create_classification_evaluator(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    state = evaluator.run(dataloader)

    assert "loss" in state.metrics
    assert "acc1" in state.metrics
    assert "acc2" in state.metrics

    assert isinstance(state.metrics["loss"], float)
    assert 0.0 <= state.metrics["acc1"] <= 1.0
    assert 0.0 <= state.metrics["acc2"] <= 1.0


def test_create_classification_evaluator_topk_only_acc1(monkeypatch):
    import imgclassvalidation.engines.classification as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "fvcore_flops",
        lambda model, example_input, device: {
            "fvcore_flops_total": 123.0,
            "fvcore_error": None,
        },
    )

    model = SimpleClassifier()
    dataloader = make_dataloader()

    config = EvalConfig(
        device=torch.device("cpu"),
        amp=False,
        non_blocking=False,
        topk=(1,),
        criterion=nn.CrossEntropyLoss(),
    )

    evaluator = create_classification_evaluator(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    state = evaluator.run(dataloader)

    assert "loss" in state.metrics
    assert "acc1" in state.metrics
    assert "acc2" not in state.metrics
    assert "acc5" not in state.metrics


def test_create_classification_evaluator_with_output_transform(monkeypatch):
    import imgclassvalidation.engines.classification as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "fvcore_flops",
        lambda model, example_input, device: {
            "fvcore_flops_total": 123.0,
            "fvcore_error": None,
        },
    )

    class DictOutputClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)

        def forward(self, x):
            return {"logits": self.linear(x)}

    model = DictOutputClassifier()
    dataloader = make_dataloader()

    config = EvalConfig(
        device=torch.device("cpu"),
        amp=False,
        non_blocking=False,
        topk=(1,),
        criterion=nn.CrossEntropyLoss(),
    )

    def output_transform(output, y):
        return output["logits"], y

    evaluator = create_classification_evaluator(
        model=model,
        dataloader=dataloader,
        config=config,
        output_transform=output_transform,
    )

    state = evaluator.run(dataloader)

    assert "loss" in state.metrics
    assert "acc1" in state.metrics
    assert 0.0 <= state.metrics["acc1"] <= 1.0


def test_create_classification_evaluator_sets_model_to_eval(monkeypatch):
    import imgclassvalidation.engines.classification as evaluator_module

    monkeypatch.setattr(
        evaluator_module,
        "fvcore_flops",
        lambda model, example_input, device: {
            "fvcore_flops_total": 123.0,
            "fvcore_error": None,
        },
    )

    model = SimpleClassifier()
    model.train()

    dataloader = make_dataloader()

    config = EvalConfig(
        device=torch.device("cpu"),
        amp=False,
        non_blocking=False,
        topk=(1,),
        criterion=nn.CrossEntropyLoss(),
    )

    evaluator = create_classification_evaluator(
        model=model,
        dataloader=dataloader,
        config=config,
    )

    evaluator.run(dataloader)

    assert model.training is False
