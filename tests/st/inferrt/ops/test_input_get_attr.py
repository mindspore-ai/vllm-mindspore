# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for torch.ops.npu.npu_transpose_batchmatmul custom call."""

import pytest
import torch
from torch import fx
import torch.nn.functional as F
import torch_npu

from ms_inferrt.torch.fx_backend import backend as fx_backend

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual


_NZ_FORMAT = 29


def _npu_transpose_batchmatmul_func(input_tensor, weight):
    """Run npu_transpose_batchmatmul with the default supported attributes."""
    return torch.ops.npu.npu_transpose_batchmatmul(input_tensor, weight)


def _npu_transpose_batchmatmul_with_perm(input_tensor, weight):
    """Run npu_transpose_batchmatmul with DSv4 attention-style permutations."""
    return torch.ops.npu.npu_transpose_batchmatmul(
        input_tensor, weight, perm_x1=(1, 0, 2), perm_y=(1, 0, 2)
    )


@torch.compiler.disable
def _disabled_linear(output_tensor, weight, first_dim):
    """Run the linear tail outside the compiled InferRT graph."""
    return F.linear(output_tensor.view(first_dim, -1), weight, None)


class _TransposeBatchMatmulLinear(torch.nn.Module):
    """DSv4 attention-style npu_transpose_batchmatmul followed by dense linear."""

    def __init__(self, group_num, head_dim_per_group, lora_rank, output_dim):
        super().__init__()
        self.wo_a_weight = torch.nn.Parameter(
            torch.empty((group_num, head_dim_per_group, lora_rank), dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.wo_b_weight = torch.nn.Parameter(
            torch.empty((output_dim, group_num * lora_rank), dtype=torch.bfloat16),
            requires_grad=False,
        )

    def forward(self, input_tensor):
        output_tensor = _npu_transpose_batchmatmul_with_perm(input_tensor, self.wo_a_weight)
        return F.linear(output_tensor.view(input_tensor.shape[0], -1), self.wo_b_weight, None)


class _TransposeBatchMatmulSplitLinear(_TransposeBatchMatmulLinear):
    """Keep npu_transpose_batchmatmul compiled and run the linear tail in eager."""

    def forward(self, input_tensor):
        output_tensor = _npu_transpose_batchmatmul_with_perm(input_tensor, self.wo_a_weight)
        return _disabled_linear(output_tensor, self.wo_b_weight, input_tensor.shape[0])


class _RepeatedTransposeBatchMatmulLinear(_TransposeBatchMatmulLinear):
    """Run two identical custom calls followed by views and dense linear ops."""

    def forward(self, input_tensor):
        first_input = input_tensor.view(input_tensor.shape[0], 8, -1)
        first_output = _npu_transpose_batchmatmul_with_perm(first_input, self.wo_a_weight)
        first_linear = F.linear(first_output.view(input_tensor.shape[0], -1), self.wo_b_weight, None)

        second_input = input_tensor.view(input_tensor.shape[0], 8, -1)
        second_output = _npu_transpose_batchmatmul_with_perm(second_input, self.wo_a_weight)
        second_linear = F.linear(second_output.view(input_tensor.shape[0], -1), self.wo_b_weight, None)

        return first_linear + second_linear


class _GetAttrNzLinear(torch.nn.Module):
    """Use self.weight as a get_attr node to reproduce NZ metadata loss."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        weight = torch.empty((output_dim, input_dim), dtype=torch.bfloat16).npu()
        weight = torch_npu.npu_format_cast_(weight, _NZ_FORMAT)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, input_tensor):
        return F.linear(input_tensor, self.weight, None)


def _resolve_fx_arg(arg, env):
    """Resolve an FX argument with values from env."""
    if isinstance(arg, fx.Node):
        return env[arg]
    if isinstance(arg, tuple):
        return tuple(_resolve_fx_arg(item, env) for item in arg)
    if isinstance(arg, list):
        return [_resolve_fx_arg(item, env) for item in arg]
    if isinstance(arg, dict):
        return {key: _resolve_fx_arg(value, env) for key, value in arg.items()}
    return arg


def _fill_example_values(gm, *inputs):
    """Populate example_value metadata for a manually traced FX graph."""
    env = {}
    input_iter = iter(inputs)
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            value = next(input_iter)
        elif node.op == "get_attr":
            value = gm
            for part in node.target.split("."):
                value = getattr(value, part)
        elif node.op == "call_function":
            args = _resolve_fx_arg(node.args, env)
            kwargs = _resolve_fx_arg(node.kwargs, env)
            value = node.target(*args, **kwargs)
        elif node.op == "call_method":
            args = _resolve_fx_arg(node.args, env)
            kwargs = _resolve_fx_arg(node.kwargs, env)
            self_obj, *rest_args = args
            value = getattr(self_obj, node.target)(*rest_args, **kwargs)
        elif node.op == "output":
            value = _resolve_fx_arg(node.args[0], env)
        else:
            raise RuntimeError(f"Unsupported FX node op: {node.op}")
        node.meta["example_value"] = value
        env[node] = value


def _format_cast_linear_weight_nz(module):
    """Cast the dense linear weight to FRACTAL_NZ for an optional control case."""
    weight_data = torch_npu.npu_format_cast_(module.wo_b_weight.data, _NZ_FORMAT)
    module.wo_b_weight = torch.nn.Parameter(weight_data, requires_grad=False)


def _make_chain_modules(group_num, head_dim_per_group, lora_rank, output_dim, use_nz_weight):
    """Create full-graph and split-linear modules with identical weights."""
    full_module = _TransposeBatchMatmulLinear(
        group_num, head_dim_per_group, lora_rank, output_dim
    ).npu()
    split_module = _TransposeBatchMatmulSplitLinear(
        group_num, head_dim_per_group, lora_rank, output_dim
    ).npu()
    torch.nn.init.normal_(full_module.wo_a_weight)
    torch.nn.init.normal_(full_module.wo_b_weight)
    split_module.wo_a_weight.data.copy_(full_module.wo_a_weight.data)
    split_module.wo_b_weight.data.copy_(full_module.wo_b_weight.data)
    if use_nz_weight:
        _format_cast_linear_weight_nz(full_module)
        _format_cast_linear_weight_nz(split_module)
    return full_module, split_module


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_npu_transpose_batchmatmul_custom_call_matches_eager():
    """
    Feature: Check npu_transpose_batchmatmul custom call
    Description: Verify InferRT falls back to custom_call and compiled output matches eager output
    Expectation: Compiled output matches eager output
    """
    input_tensor = torch.randn((2, 4, 16), dtype=torch.float16).npu()
    weight = torch.randn((2, 16, 16), dtype=torch.float16).npu()

    expected = _npu_transpose_batchmatmul_func(input_tensor, weight).detach().cpu()

    torch.compiler.reset()
    compiled_func = torch.compile(_npu_transpose_batchmatmul_func, backend=fx_backend, fullgraph=True)
    actual = compiled_func(input_tensor, weight).detach().cpu()
    torch.npu.synchronize()

    AssertRtolEqual(expected, actual)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_repeated_npu_transpose_batchmatmul_linear_chain_matches_eager():
    """
    Feature: Check repeated npu_transpose_batchmatmul followed by dense linear
    Description: Verify two identical custom calls in one graph and repeated graph execution
    Expectation: Both executions match eager output
    """
    torch.manual_seed(0)
    batch_seq = 128
    group_num = 8
    head_dim_per_group = 4096
    lora_rank = 1024
    output_dim = 4096

    module = _RepeatedTransposeBatchMatmulLinear(
        group_num, head_dim_per_group, lora_rank, output_dim
    ).npu()
    torch.nn.init.normal_(module.wo_a_weight)
    torch.nn.init.normal_(module.wo_b_weight)

    first_input = torch.randn(
        (batch_seq, group_num, head_dim_per_group), dtype=torch.bfloat16
    ).npu()
    second_input = torch.randn(
        (batch_seq, group_num, head_dim_per_group), dtype=torch.bfloat16
    ).npu()

    first_expected = module(first_input).detach().cpu()
    second_expected = module(second_input).detach().cpu()

    torch.compiler.reset()
    compiled_module = torch.compile(module, backend=fx_backend, fullgraph=True)
    first_actual = compiled_module(first_input).detach().cpu()
    second_actual = compiled_module(second_input).detach().cpu()
    torch.npu.synchronize()

    AssertRtolEqual(first_expected, first_actual)
    AssertRtolEqual(second_expected, second_actual)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_get_attr_nz_linear_reproduces_format_metadata_loss():
    """
    Feature: Reproduce get_attr NZ weight metadata loss
    Description: Trace self.weight as get_attr and compare InferRT linear with eager linear
    Expectation: Current get_attr conversion loses FRACTAL_NZ metadata and produces wrong output
    """
    torch.manual_seed(0)
    input_dim = 8192
    output_dim = 4096
    batch = 128

    module = _GetAttrNzLinear(input_dim, output_dim).npu()
    torch.nn.init.normal_(module.weight)
    input_tensor = torch.randn((batch, input_dim), dtype=torch.bfloat16).npu()

    assert int(torch_npu.get_npu_format(module.weight)) == _NZ_FORMAT
    expected = module(input_tensor).detach().cpu()

    gm = fx.symbolic_trace(module)
    get_attr_nodes = [node for node in gm.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) == 1
    _fill_example_values(gm, input_tensor)

    compiled_module = fx_backend(gm, [input_tensor])
    actual = compiled_module(input_tensor).detach().cpu()
    torch.npu.synchronize()

    AssertRtolEqual(expected, actual)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("use_nz_weight", [False, True])
def test_npu_transpose_batchmatmul_linear_chain_matches_eager(use_nz_weight):
    """
    Feature: Check npu_transpose_batchmatmul followed by dense linear
    Description: Compare full InferRT graph and split-linear graph with eager execution
    Expectation: TBMM custom call and following InferRT linear both match eager output
    """
    torch.manual_seed(0)
    batch_seq = 128
    group_num = 8
    head_dim_per_group = 4096
    lora_rank = 1024
    output_dim = 4096

    full_module, split_module = _make_chain_modules(
        group_num, head_dim_per_group, lora_rank, output_dim, use_nz_weight
    )
    input_tensor = torch.randn(
        (batch_seq, group_num, head_dim_per_group), dtype=torch.bfloat16
    ).npu()

    expected = full_module(input_tensor).detach().cpu()

    torch.compiler.reset()
    compiled_full = torch.compile(full_module, backend=fx_backend, fullgraph=True)
    actual_full = compiled_full(input_tensor).detach().cpu()
    torch.npu.synchronize()
    AssertRtolEqual(expected, actual_full)

    torch.compiler.reset()
    compiled_split = torch.compile(split_module, backend=fx_backend, fullgraph=False)
    actual_split = compiled_split(input_tensor).detach().cpu()
    torch.npu.synchronize()
    AssertRtolEqual(expected, actual_split)
