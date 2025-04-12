import torch

def apply_temperature(
    self,
    logits: torch.Tensor,
    temp: torch.Tensor,
) -> torch.Tensor:
    # logits.div_ will cause some error right now.
    # So we use logits = logits.div instead of logits.div_.
    return logits.div(temp.unsqueeze(dim=1))
