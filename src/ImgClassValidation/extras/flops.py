from typing import Dict, Optional

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


@torch.no_grad()
def fvcore_flops(
    model: nn.Module, example_input: torch.Tensor, device: torch.device
) -> Dict[str, Optional[float]]:
    if FlopCountAnalysis is None:
        return {"fvcore_flops_total": None, "fvcore_error": "fvcore_not_installed"}

    try:
        model.eval()
        x = example_input.to(device)
        flops = FlopCountAnalysis(model, (x,))
        return {"fvcore_flops_total": float(flops.total()), "fvcore_error": None}
    except Exception as e:
        return {"fvcore_flops_total": None, "fvcore_error": repr(e)}
