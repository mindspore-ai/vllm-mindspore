import torch
from mrt.ir import Tensor

import _mrt_torch

def from_torch(torch_tensor: torch.Tensor) -> Tensor:
    return _mrt_torch.from_torch(torch_tensor)

def to_torch(tensor: Tensor) -> torch.Tensor:
    return _mrt_torch.to_torch(tensor)

def update_tensor_data(tensor: Tensor, torch_tensor: torch.Tensor):
    _mrt_torch.update_tensor_data(tensor, torch_tensor)
