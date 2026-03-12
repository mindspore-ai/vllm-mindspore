# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Tests for setitem operation with torch dynamo."""
import torch
from torch.fx.experimental.proxy_tensor import make_fx

def my_compiler(gm, example_inputs):
    """Debug compiler that prints FX graphs and returns gm.forward."""
    print("--- FX Graph Nodes ---")
    gm.graph.print_tabular()

    for node in gm.graph.nodes:
        print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}")
        example_value = node.meta.get("example_value", None)
        print(f"  Meta example_value: {example_value}")

    print()
    print("--- start make fx for dynamo gm ---")
    gm2 = make_fx(gm)(*example_inputs)
    print("--- end make fx for dynamo gm ---")
    print("--- start print make fx for dynamo gm ---")
    print(gm2.graph)
    print("--- end print make fx for dynamo gm ---")
    print()

    return gm.forward  # Return forward to continue execution

def test_setitem_with_tensor_index_dynamo():
    """
    Feature: Test setitem with tensor index using torch dynamo
    Description: Test setitem where index is a tensor from operator output, compiled with torch dynamo
    Expectation: The result is correct
    """
    def func(x, idx_tensor, value):
        x[idx_tensor] = value
        return x
    compiled_op = torch.compile(func, backend=my_compiler)
    x = torch.randn(4, 2, 3).cuda()
    idx_tensor = torch.tensor([0, 2]).cuda()
    value = torch.randn(2, 2, 3).cuda()
    out = compiled_op(x, idx_tensor, value)
    print(out)

def test_setitem_with_tensor_index_make_fx():
    """
    Feature: Test setitem with tensor index using make_fx
    Description: Test setitem where index is a tensor from operator output, captured with make_fx
    Expectation: The result is correct
    """
    def func(x, idx_tensor, value):
        res = x.clone()
        res[idx_tensor] = value
        res = res * 2
        return res

    x = torch.randn(4, 2, 3).cuda()
    idx_tensor = torch.tensor([0, 2]).cuda()
    value = torch.randn(2, 2, 3).cuda()

    print("======================make_fx graph======================")
    gm = make_fx(func)(x, idx_tensor, value)
    print("--- FX Graph ---")
    print(gm.graph)
    print("--- FX Graph Nodes ---")
    gm.graph.print_tabular()

    for node in gm.graph.nodes:
        print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}, Args: {node.args}")
        val = node.meta.get("val", None)
        print(f"  Meta val: {val}")

    out = gm(x, idx_tensor, value)
    expected = func(x, idx_tensor, value)
    print("Output:", out)
    print("Expected:", expected)
