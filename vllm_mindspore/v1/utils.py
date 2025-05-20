import numpy as np
import torch
import mindspore as ms

def _copy_slice_from_np(from_np: np.ndarray, to_tensor: torch.Tensor,
                        length: int) -> None:
    """
    Copy the first length elements of a numpy array into a tensor in a
    non-blocking manner.
    """
    to_tensor[:length] = ms.from_numpy(from_np[:length])
    return to_tensor


def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor,
               length: int, *, return_tensor=True) -> None:
    """
    Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.
    """
    to_tensor[:length] = from_tensor[:length]
    if return_tensor:
        return to_tensor[:length]
