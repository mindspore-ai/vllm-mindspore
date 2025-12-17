import pytest
import numpy as np
import torch

from tests.mark_utils import arg_mark
from tests.ops_utils import AssertRtolEqual
from mrt.torch import fx_mlir_backend as backend


def expand_op(x, y):
    """op function for expand with dynamic size based on x.size"""
    # Use size(int) so that aten.size.int is in the graph
    b = x.size(0)
    target_shape = (b, 16)
    return torch.broadcast_to(y, target_shape)


def get_op_func_compiled():
    return torch.compile(expand_op, backend=backend)


# 去掉 shape 的 parametrize，改在内部循环
@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize("pipeline", (True, False))
@pytest.mark.parametrize("shape", [(2, 8), (4, 8)])
def test_expand(pipeline, monkeypatch, shape):
    """
    Feature: Test aclnn expand
    Description: Test aclnn expand with dynamic size based on x.size
    Expectation: The result is correct
    """

    if pipeline:
        monkeypatch.setenv("MRT_ENABLE_PIPELINE", "on")

    compile_op = get_op_func_compiled()
    prec = 1e-4

    cpu_input0 = np.random.uniform(-1, 1, shape).astype(np.float32)
    cpu_input1 = np.random.uniform(-1, 1, (1, 1)).astype(np.float32)
    cpu_tensor0 = torch.from_numpy(cpu_input0)
    cpu_tensor1 = torch.from_numpy(cpu_input1)
    npu_tensor0 = torch.from_numpy(cpu_input0).npu()
    npu_tensor1 = torch.from_numpy(cpu_input1).npu()

    cpu_output = expand_op(cpu_tensor0, cpu_tensor1).detach().cpu().numpy()
    npu_output = compile_op(npu_tensor0, npu_tensor1).detach().cpu().numpy()

    AssertRtolEqual(cpu_output, npu_output, prec)
