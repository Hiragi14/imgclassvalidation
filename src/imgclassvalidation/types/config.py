from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class EvalConfig:
    """
    Configuration container for classification model evaluation.

    This dataclass defines the *minimal and stable API boundary* for controlling
    evaluation behavior in this library. It is intentionally immutable
    (``frozen=True``) to guarantee reproducibility and to avoid side effects
    during evaluation.

    Attributes:
        device (torch.device):
            Device on which evaluation is executed.
            Typical values are ``torch.device("cuda")`` or
            ``torch.device("cpu")``.

            All input tensors and the model are expected to be moved to this
            device prior to or during evaluation.

        amp (bool, default=False):
            Whether to enable Automatic Mixed Precision (AMP) during evaluation.

            If ``True``, model inference is wrapped with ``torch.autocast``.
            This is typically enabled when using CUDA to reduce memory usage
            and improve throughput. Has no effect on CPU unless explicitly
            supported.

        non_blocking (bool, default=True):
            Whether to use non-blocking memory transfers when moving tensors
            to the target device.

            This flag is effective only when:
            - using CUDA
            - the DataLoader is configured with ``pin_memory=True``

        topk (tuple[int, ...], default=(1, 5)):
            Top-k values for accuracy computation.

            For example:
            - ``(1,)`` computes top-1 accuracy only
            - ``(1, 5)`` computes top-1 and top-5 accuracy (ImageNet-style)

            Each value ``k`` results in a metric named ``acc{k}``
            (e.g., ``acc1``, ``acc5``).

        criterion (torch.nn.Module | None, default=None):
            Loss function used for evaluation.

            If ``None``, ``torch.nn.CrossEntropyLoss`` is used by default.
            This parameter allows users to supply a custom criterion
            (e.g., label smoothing, class-weighted loss) without modifying
            the evaluation loop.
    """

    device: torch.device
    amp: bool = False
    non_blocking: bool = True
    topk: tuple[int, ...] = (1, 5)
    criterion: nn.Module | None = None


@dataclass
class EvalResult:
    """
    Container for evaluation results returned by the public evaluation API.

    This dataclass separates *core evaluation metrics* from *auxiliary
    statistics* to keep the API extensible and stable as new reporting features
    are added.

    Attributes:
        metrics (dict[str, float]):
            Aggregated evaluation metrics computed over the entire dataset.

            Typical entries include:
            - ``loss`` : average evaluation loss
            - ``acc1`` : top-1 accuracy
            - ``acc5`` : top-5 accuracy (if enabled)

            All values are converted to native Python ``float`` for ease of
            logging, serialization, and downstream processing.

        extras (dict[str, Any]):
            Auxiliary evaluation information computed once per evaluation run.

            This dictionary is intentionally flexible and may contain:
            - Model statistics:
                - ``params_total``
                - ``params_trainable``
            - Computational complexity:
                - ``fvcore_macs_total``
                - ``fvcore_flops_total`` (derived)
            - Human-readable strings (optional):
                - ``fvcore_macs_str``
                - ``fvcore_flops_str``

            The exact contents depend on which optional reporters or utilities
            are enabled, and new keys may be added without breaking the API.
    """

    metrics: dict[str, float]
    extras: dict[str, Any]
