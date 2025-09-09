import torch
from typing import Any
from mrt.ir import Value, Tensor, Tuple

import mrt._mrt_torch as _mrt_torch


def from_torch(obj: Any) -> Value:
    if isinstance(obj, Value):
        return obj
    if isinstance(obj, (list, tuple)):
        return Value(Tuple([from_torch(e) for e in obj]))
    if isinstance(obj, torch._subclasses.FakeTensor):
        return Value(_mrt_torch.from_torch(obj, is_fake=True))
    if isinstance(obj, torch.Tensor):
        return Value(_mrt_torch.from_torch(obj))
    if isinstance(obj, (int, float, bool, str)):
        return Value(obj)
    if obj is None:
        return Value()
    raise TypeError(f"Unsupported python type for conversion to mrt.ir.Value: {type(obj)}")


def to_torch(value: Value) -> Any:
    if not isinstance(value, Value):
        return value
    if value.is_none():
        return None
    if value.is_tensor():
        return _mrt_torch.to_torch(value.to_tensor())
    if value.is_tuple():
        return tuple(to_torch(item) for item in value.to_tuple())
    if value.is_int():
        return value.to_int()
    if value.is_double():
        return value.to_double()
    if value.is_bool():
        return value.to_bool()
    if value.is_string():
        return value.to_string()
    raise TypeError(f"Unsupported mrt.ir.Value for conversion to python object: {value}")


def update_tensor_data(tensor: Tensor, torch_tensor: torch.Tensor):
    _mrt_torch.update_tensor_data(tensor, torch_tensor)
