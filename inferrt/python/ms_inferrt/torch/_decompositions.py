# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file is derived from:
# https://github.com/iree-org/iree-turbine/blob/main/iree/turbine/dynamo/decompositions.py

"""Decomposition utilities and context management for PyTorch ops.

This module manages decomposition tables for different compilation scopes,
providing:
- A thread-local stack of decomposition tables per scope.
- Helpers to extend or prune decompositions within a context.
- A default set of decompositions collected from torch._decomp.
"""

from typing import Sequence, Union
import contextlib
import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

# pylint: disable=protected-access

DecompositionOpsList = Sequence[
    Union[torch._ops.OperatorBase, torch._ops.OpOverloadPacket]
]


def _get_default_decomposition_ops() -> DecompositionOpsList:
    """Collects the default set of operator decompositions used by this module."""
    aten = torch.ops.aten
    # default decompositions pulled from SHARK / torch._decomp
    return [
        aten._adaptive_avg_pool2d,
        aten._euclidean_dist,
        aten._log_softmax,
        aten._log_softmax_backward_data,
        aten._native_batch_norm_legit,
        aten._native_batch_norm_legit.no_stats,
        aten._native_batch_norm_legit_functional,
        aten._native_batch_norm_legit_no_training,
        aten._prelu_kernel,
        aten._to_copy,
        aten._unsafe_index.Tensor,
        aten.addmm,
        aten.binary_cross_entropy,
        aten.dot,
        aten.embedding_dense_backward,
        aten.full,
        aten.grid_sampler_2d,
        aten.im2col,
        aten.index_copy,
        aten.index_copy_,
        aten.lift_fresh_copy.default,
        aten.linspace.Tensor_Tensor,
        aten.log_sigmoid_forward,
        aten.masked_fill.Scalar,
        aten.masked_fill.Tensor,
        aten.native_group_norm,
        aten.native_layer_norm,
        aten.native_layer_norm_backward,
        aten.nll_loss_backward,
        aten.nll_loss_forward,
        aten.norm.ScalarOpt_dim,
        aten.select_backward,
        aten.slice_backward,
        aten.soft_margin_loss,
        aten.squeeze.dims,
        aten.t,
        aten.unbind.int,
        aten.unsafe_split.Tensor,
        aten.upsample_bilinear2d.vec,
    ]


def _contains_symint(example_inputs) -> bool:
    """Check if any of the example inputs contains a torch.SymInt.

    Args:
        example_inputs: A sequence of example inputs to check.

    Returns:
        True if any input or its shape contains a torch.SymInt, False otherwise.
    """
    for arg in example_inputs:
        if isinstance(arg, torch.SymInt):
            return True
        if isinstance(arg, torch.Tensor):
            for s in arg.shape:
                if isinstance(s, torch.SymInt):
                    return True
    return False


def apply_decompositions(gm: torch.fx.GraphModule, example_inputs):
    """Apply operator decompositions to a GraphModule if requested.

    Note:
        The torch._ops types are part of PyTorch's operator registry surface.
        We suppress Pylint's protected-access warning for the type annotation.
    """
    decompositions = get_decompositions(_get_default_decomposition_ops())

    # 1. Try to reuse an existing FakeTensorMode from tensor inputs (typical case)
    fake_mode = next(
        (arg.fake_mode for arg in example_inputs if isinstance(arg, FakeTensor)),
        None
    )

    # 2. If none is found (e.g., only SymInt inputs), rebuild ShapeEnv from SymInt and create a mode
    if fake_mode is None:
        shape_env = None
        for arg in example_inputs:
            if isinstance(arg, torch.SymInt):
                # Grab the node behind the SymInt and its ShapeEnv
                node = getattr(arg, "node", None)
                shape_env = getattr(node, "shape_env", None)
                if shape_env:
                    break

        # If ShapeEnv is available, create a FakeTensorMode sharing it
        # This lets torch.empty create FakeTensor in the correct symbolic context
        if shape_env:
            fake_mode = FakeTensorMode(shape_env=shape_env)

    # 3. Activate fake_mode and use tracing_mode="real"
    # "real": avoid make_fx creating a new mode that conflicts
    # with fake_mode: ensure factory ops (e.g., empty) dispatch to FakeTensor impl
    with fake_mode if fake_mode else contextlib.nullcontext():
        gm = make_fx(
            functionalize(gm),
            decomposition_table=decompositions,
            tracing_mode="real",
        )(*example_inputs)
    return gm
