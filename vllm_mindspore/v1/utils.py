import torch

def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor,
               length: int) -> None:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.
    """
    to_tensor[:length] = from_tensor[:length]
    return to_tensor
