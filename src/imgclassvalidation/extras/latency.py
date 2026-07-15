import numpy as np
import torch
from torch import nn


def measure_cuda_latency(
    model: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device | None = None,
    *,
    warmup_steps: int = 50,
    measurement_steps: int = 100,
) -> dict[str, float | None | str]:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        if device is None:
            device = torch.device("cuda")

        model = model.to(device)
        model.eval()

        inputs = input_tensor.to(device)

        with torch.inference_mode():
            for _ in range(warmup_steps):
                _ = model(inputs)

        torch.cuda.synchronize()

        latencies_ms = []

        with torch.inference_mode():
            for _ in range(measurement_steps):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                _ = model(inputs)
                end.record()

                end.synchronize()
                latencies_ms.append(start.elapsed_time(end))

        latency_array = np.asarray(latencies_ms)

        return {
            "mean_ms": float(latency_array.mean()),
            "median_ms": float(np.median(latency_array)),
            "std_ms": float(latency_array.std()),
            "min_ms": float(latency_array.min()),
            "p90_ms": float(np.percentile(latency_array, 90)),
            "p95_ms": float(np.percentile(latency_array, 95)),
            "p99_ms": float(np.percentile(latency_array, 99)),
            "max_ms": float(latency_array.max()),
            "latency_error": None,
        }
    except Exception as e:
        return {
            "mean_ms": None,
            "median_ms": None,
            "std_ms": None,
            "min_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "latency_error": repr(e),
        }
