"""Implementations for torch getitem operation processing."""

from typing import Tuple
import torch
from torch.fx.node import Argument, Node
from mrt.ir import Op
from mrt.torch.utils import tuple_indices_to_slice_arg


def _tensor_getitem_by_tuple(x: Node, indices: Tuple[Argument, ...]):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1, ..., 1:10:2] -> getitem_slice(x, (1, ellipsis, slice(1, 10, 2)))
    shape = x.meta["example_value"].shape
    begin, end, steps, axes = tuple_indices_to_slice_arg(indices, shape)
    return Op.getitem_slice, [x, begin, end, axes, steps]

def _tensor_getitem_by_slice(x: Node, indices: slice):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1:10:2] -> getitem_slice(x, slice(1, 10, 2))
    shape = x.meta["example_value"].shape
    start = indices.start if indices.start is not None else 0
    end = indices.stop if indices.stop is not None else shape[0]
    step = indices.step if indices.step is not None else 1
    return Op.getitem_slice, [x, [start], [end], [0], [step]]

def _tensor_getitem_by_number(x: Node, indices: int):
    """Convert getitem indices to getitem_slice parameters."""
    # x[1] -> getitem_slice(x, [1], [2], [0], [1])
    shape = x.meta["example_value"].shape
    starts = [indices]
    ends = [indices + 1]
    axes = [0]
    steps = [1]
    for idx, end in enumerate(shape[1:]):
        starts.append(0)
        ends.append(end)
        axes.append(idx + 1)
        steps.append(1)
    return Op.getitem_slice, [x, starts, ends, axes, steps]

def _tensor_getitem_by_tensor(x: Node, indices: torch.Tensor):
    """Convert getitem indices to gather_v2 parameters."""
    # x[tensor] -> gather_v2(x, 0, tensor)
    return Op.gather_v2, [x, 0, indices]

def tuple_getitem(x, indices):
    """Handle tuple_getitem node."""
    # operation: Op.tuple_getitem(x, indices)
    return Op.tuple_getitem, [x, indices]

# pylint: disable=unused-argument
def getitem_process(node, input_nodes):
    """Handle getitem node."""
    if isinstance(input_nodes[0], (list, tuple)):
        return tuple_getitem(input_nodes[0], input_nodes[1])

    if isinstance(input_nodes[0], torch.fx.node.Node) and \
       isinstance(input_nodes[0].meta.get("example_value"), (list, tuple)):
        return tuple_getitem(input_nodes[0], input_nodes[1])

    # input is tensor
    is_tensor_node = isinstance(input_nodes[0], torch.fx.node.Node)
    example_val = input_nodes[0].meta.get("example_value") if is_tensor_node else None
    # pylint: disable=protected-access
    if is_tensor_node and (input_nodes[0].type == torch.Tensor or
                           isinstance(example_val, torch._subclasses.FakeTensor)):
        idx_type = type(input_nodes[1])
        if idx_type is int:
            return _tensor_getitem_by_number(input_nodes[0], input_nodes[1])
        if idx_type is slice:
            return _tensor_getitem_by_slice(input_nodes[0], input_nodes[1])
        if idx_type is tuple:
            return _tensor_getitem_by_tuple(input_nodes[0], input_nodes[1])
        if isinstance(input_nodes[1], torch.fx.node.Node) and input_nodes[0].type == torch.Tensor:
            return _tensor_getitem_by_tensor(input_nodes[0], input_nodes[1])
        raise ValueError(f"Unsupported getitem indices type: {idx_type}")
    raise ValueError(f"Unsupported getitem input type: {type(input_nodes[0])}")
