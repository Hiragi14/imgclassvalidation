from collections.abc import Callable

import torch
import torch.nn as nn
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy
from rich.console import Console
from torch.utils.data import DataLoader

from imgclassvalidation.extras import count_params, fvcore_flops, measure_cuda_latency
from imgclassvalidation.hooks import attach_rich_progress
from imgclassvalidation.types import EvalConfig, EvalResult

console_ = Console()


def _default_criterion():
    return nn.CrossEntropyLoss()


# -------------------------
# Core: create evaluator
# -------------------------
def create_classification_evaluator(
    model: nn.Module,
    dataloader: DataLoader,
    config: EvalConfig,
    *,
    output_transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
) -> Engine:
    """
    Ignite evaluator (classification).
    - dataloader: yields (x, y)
    - model(x) -> logits
    - engine.state.output: (logits, y) by default

    output_transform:
      (logits, y) -> (logits, y) に変換するフック
      例: modelがdictを返す、label整形が必要、などの吸収に使う
    """

    device = config.device
    criterion = config.criterion if config.criterion is not None else _default_criterion()

    def _inference(engine: Engine, batch: tuple[torch.Tensor, torch.Tensor]):
        model.eval()
        x, y = batch
        x = x.to(device, non_blocking=config.non_blocking)
        y = y.to(device, non_blocking=config.non_blocking)

        with torch.autocast(device_type=device.type, enabled=config.amp):
            logits = model(x)

        if output_transform is not None:
            logits, y = output_transform(logits, y)

        return logits, y

    evaluator = Engine(_inference)

    # --- metrics ---
    Loss(criterion, output_transform=lambda out: (out[0], out[1])).attach(evaluator, "loss")

    if 1 in config.topk:
        Accuracy(output_transform=lambda out: (out[0], out[1])).attach(evaluator, "acc1")
    for k in config.topk:
        if k != 1:
            TopKCategoricalAccuracy(k=k, output_transform=lambda out: (out[0], out[1])).attach(
                evaluator, f"acc{k}"
            )

    # --- extras: store once at STARTED ---
    evaluator.state.extras = {}  # ty: ignore[unresolved-attribute]

    @evaluator.on(Events.STARTED)
    def _on_started(engine: Engine):
        engine.state.extras.update(count_params(model))  # ty: ignore[unresolved-attribute]
        # ImageNet1k前提: 大規模データでもFLOPsは1回だけ推定する
        # example_input は loader先頭から1枚取る（ユーザーが明示入力を渡す拡張も後で可能）
        x0, _ = next(iter(dataloader))
        example = x0[:1].contiguous()
        engine.state.extras.update(fvcore_flops(model, example, device))  # ty: ignore[unresolved-attribute]
        # Measure latency
        engine.state.extras.update(measure_cuda_latency(model, example, device))  # ty: ignore[unresolved-attribute]

    return evaluator


# -------------------------
# Public API: run evaluation
# -------------------------
@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    dataloader: DataLoader,
    config: EvalConfig,
    *,
    output_transform: Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    | None = None,
    progress: bool = True,
    progress_type: str = "rich",
    progress_metrics: tuple[str, ...] = ("loss", "acc1", "acc5"),
    console: Console = console_,
) -> EvalResult:
    """
    Run classification model evaluation using an Ignite-based evaluator.

    This function performs a single-pass evaluation of a classification model
    on a given evaluation DataLoader and returns aggregated metrics together
    with auxiliary statistics (e.g., parameter count, FLOPs).

    Ignite is used internally to construct the evaluation loop, but is treated
    as an implementation detail and is not exposed through the public API.

    Params:
        model (torch.nn.Module):
            Classification model to be evaluated.
            The model must accept a tensor input ``x`` and return classification
            logits of shape ``(N, C)``.

        dataloader (torch.utils.data.DataLoader):
            **Evaluation DataLoader** (validation or test).
            Each batch must yield a tuple ``(inputs, targets)``, where:

            - ``inputs`` (torch.Tensor):
                Input images of shape ``(N, C, H, W)``.
            - ``targets`` (torch.Tensor):
                Ground-truth class indices of shape ``(N,)``.

            Notes:
            - This DataLoader is assumed to be read-only and used only for
              evaluation.
            - Training-time data augmentation and label manipulation should
              NOT be applied.
            - Shuffling should be disabled except when required by distributed
              samplers.

        config (EvalConfig):
            Evaluation configuration specifying device placement, AMP usage,
            top-k metrics, and other evaluation options.

        output_transform (callable, optional):
            Optional post-processing function applied to the raw model outputs.

            The callable must have the signature::

                output_transform(logits, targets) -> (logits, targets)

            This hook can be used to:
            - Extract logits from dictionary-based model outputs
            - Modify or reshape targets
            - Apply task-specific output normalization

            If ``None`` (default), the raw ``(logits, targets)`` pair is used.

    Returns:
        EvalResult:
            Evaluation result container with the following fields:

            - ``metrics`` (dict[str, float]):
                Aggregated evaluation metrics such as loss, top-1 accuracy,
                and top-k accuracy.

            - ``extras`` (dict[str, Any]):
                Auxiliary evaluation statistics computed once per run, including
                parameter count, sparsity metrics, and FLOPs (if enabled).

    Notes:
        - This function performs evaluation only; training and gradient updates
          are not supported.
        - Distributed evaluation (DDP metric reduction) is intentionally
          excluded from this core API and should be implemented in a
          higher-level wrapper.
        - Visualization, confusion matrices, misclassification analysis,
          and experiment logging (e.g., W&B, TensorBoard) are designed to be
          added as modular extensions.
    """
    evaluator = create_classification_evaluator(
        model=model,
        dataloader=dataloader,
        config=config,
        output_transform=output_transform,
    )

    if progress:
        if progress_type == "rich":
            attach_rich_progress(
                evaluator,
                total=len(dataloader),
                description="Evaluating",
                console=console,
            )
        else:
            ProgressBar(persist=True).attach(
                evaluator,
                metric_names=list(progress_metrics),
            )

    evaluator.run(dataloader)

    metrics = {k: float(v) for k, v in evaluator.state.metrics.items()}
    extras = dict(getattr(evaluator.state, "extras", {}))
    return EvalResult(metrics=metrics, extras=extras)
