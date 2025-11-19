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
import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.func import functionalize

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
        aten.split.Tensor,
        aten.split_with_sizes,
        aten.squeeze.dims,
        aten.t,
        aten.unbind.int,
        aten.unsafe_split.Tensor,
        aten.upsample_bilinear2d.vec,
    ]

def apply_decompositions(gm: torch.fx.GraphModule, example_inputs):
    """Apply operator decompositions to a GraphModule if requested.

    Note:
        The torch._ops types are part of PyTorch's operator registry surface.
        We suppress Pylint's protected-access warning for the type annotation.
    """
    decompositions = get_decompositions(_get_default_decomposition_ops())
    gm = make_fx(
        functionalize(gm),
        decomposition_table=decompositions,
    )(*example_inputs)
    return gm
