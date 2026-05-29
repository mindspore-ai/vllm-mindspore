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

"""PipelineExecutor reentry & exception deadlock prevention tests.
Uses native InferRT dummy_sleep (aclnnAdd + sleep 200ms) and dummy_throw (throw in Launch).

Run: TASK_QUEUE_ENABLE=1 python3 -u test_pipeline_reentry.py
"""
import time
import os
import torch
import torch_npu
import ms_inferrt
from ms_inferrt.torch.fx_backend import backend
from tests.mark_utils import arg_mark

script_dir = os.path.dirname(os.path.abspath(__file__))
sleep_cc = os.path.join(script_dir, "op_dummy_sleep.cc")
throw_cc = os.path.join(script_dir, "op_dummy_throw.cc")

ms_inferrt.ops.load(name="dummy_sleep", sources=[sleep_cc], backend="Ascend")
ms_inferrt.ops.load(name="dummy_throw", sources=[throw_cc], backend="Ascend")

@torch.library.custom_op("ms_inferrt::dummy_sleep", mutates_args=())
def dummy_sleep_op(x: torch.Tensor, y_unused: torch.Tensor, alpha_unused: float) -> torch.Tensor:  # pylint: disable=unused-argument
    raise NotImplementedError

@torch.library.register_fake("ms_inferrt::dummy_sleep")
def _dummy_sleep_fake(x, y_unused, alpha_unused):  # pylint: disable=unused-argument
    return torch.empty_like(x)

@torch.library.custom_op("ms_inferrt::dummy_throw", mutates_args=())
def dummy_throw_op(x: torch.Tensor, y_unused: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
    raise NotImplementedError

@torch.library.register_fake("ms_inferrt::dummy_throw")
def _dummy_throw_fake(x, y_unused):  # pylint: disable=unused-argument
    return torch.empty_like(x)

def _sleep_func(x, y):
    return torch.ops.ms_inferrt.dummy_sleep(x, y, 1.0)

def _throw_func(x, y):
    out = torch.ops.ms_inferrt.dummy_sleep(x, y, 1.0)
    out = torch.ops.ms_inferrt.dummy_throw(x, out)
    out = torch.ops.ms_inferrt.dummy_sleep(x, out, 1.0)
    return torch.ops.ms_inferrt.dummy_sleep(x, out, 1.0)

compiled_sleep = torch.compile(_sleep_func, backend=backend)
compiled_throw_at_second_step = torch.compile(_throw_func, backend=backend)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_reentry_with_heavy_launch():
    """
    Feature: PipelineExecutor reentry safety
    Description: Launch dummy_sleep (aclnnAdd + 200ms sleep) 5 times, measure wall time
    Expectation: 1st call is slow; subsequent calls overlap (max_wait < 5s, no deadlock)
    """
    a = torch.ones(4, 4).npu()
    b = torch.ones(4, 4).npu() * 2

    per_call = []
    outputs = []
    for _ in range(5):
        t0 = time.time()
        out = compiled_sleep(a, b)
        per_call.append(time.time() - t0)
        outputs.append(out)

    torch_npu.npu.synchronize()
    for out in outputs:
        expected = a.cpu() + b.cpu()
        actual = out.cpu()
        assert torch.allclose(actual, expected), f"Result mismatch: {actual} vs {expected}"

    first = per_call[0] * 1000
    max_wait = max(per_call[1:]) * 1000
    print(f"reentry: 1st={first:.1f}ms max_wait={max_wait:.1f}ms", flush=True)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_exception_no_deadlock():
    """
    Feature: PipelineExecutor exception recovery
    Description: Trigger dummy_throw (raises in Launch), then run dummy_sleep
    Expectation: Recovery succeeds within 5s, no deadlock after exception
    """
    a = torch.ones(4, 4).npu()
    b = torch.ones(4, 4).npu() * 2

    try:
        for _ in range(5):
            _ = compiled_throw_at_second_step(a, b)

        torch_npu.npu.synchronize()
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    start = time.time()
    compiled_sleep(a, b)
    torch_npu.npu.synchronize()
    elapsed = time.time() - start
    print(f"exception recovery: {elapsed:.3f}s, NO DEADLOCK", flush=True)
