#!/usr/bin/env python3
# encoding: utf-8
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
# ============================================================================

import contextlib
import gc
import logging
import os
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch

if TYPE_CHECKING:
    from torch.library import Library
else:
    Library = None

from vllm.utils import T, TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad

import mindspore as ms
from mindspore.common.initializer import Zero
from mindspore import dtype as mstype

from .scripts import env_setup

MsKVCache = Tuple[ms.Tensor, ms.Tensor]

logger = logging.getLogger(__name__)


STR_DTYPE_TO_MS_DTYPE = {
    "half": ms.float16,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float": ms.float32,
    "fp8": ms.uint8,
    "fp8_e4m3": ms.uint8,
    "fp8_e5m2": ms.uint8,
}


def get_valid_dtype(dtype):
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_MS_DTYPE[dtype]
    return dtype


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
    dispatch_key: str = "CUDA",
): ...


@contextlib.contextmanager
def memory_profiling(
    baseline_memory_in_bytes: int, weights_memory_in_bytes: int
) -> Generator["MemoryProfilingResult", None, None]:
    """Memory profiling context manager.
    baseline_memory_in_bytes: memory used by all the components other than
        the current vLLM instance. It contains: memory used by other processes, memory
        used by another vLLM instance in the same process, etc. It is usually measured
        before the current vLLM instance initialize the device. And we assume it is
        constant during the profiling of the current vLLM instance.
    weights_memory_in_bytes: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory_in_bytes because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e. before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the argument `weights_memory_in_bytes`.

    The increase of ``torch.cuda.memory_stats()["allocated_bytes.all.peak"]` after profiling gives (b.).

    (c.) is tricky. We measure the total memory used in this GPU (`torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]`),
    subtract the baseline memory, the memory used by the model weights, and diff of `torch.cuda.memory_stats()["allocated_bytes.all.current"]`.
    """  # noqa
    torch.cuda.reset_peak_memory_stats()

    from vllm.utils import MemoryProfilingResult

    result = MemoryProfilingResult()

    result.baseline_memory_in_bytes = baseline_memory_in_bytes
    # the part of memory used for holding the model weights
    result.weights_memory_in_bytes = weights_memory_in_bytes

    result.before_profile.measure()

    yield result

    gc.collect()
    torch.cuda.empty_cache()

    result.after_profile.measure()

    diff = result.after_profile - result.before_profile
    result.torch_peak_increase_in_bytes = diff.torch_peak_in_bytes

    # For mindspore, the memory is allocated and free in memory pool, so cannot read the current used memory by `torch.cuda.mem_get_info`.
    current_cuda_memory_bytes = result.after_profile.torch_memory_in_bytes
    result.non_torch_increase_in_bytes = (
        current_cuda_memory_bytes
        - baseline_memory_in_bytes
        - weights_memory_in_bytes
        - diff.torch_memory_in_bytes
    )  # noqa
    result.profile_time = diff.timestamp
    result.non_kv_cache_memory_in_bytes = (
        result.non_torch_increase_in_bytes
        + result.torch_peak_increase_in_bytes
        + result.weights_memory_in_bytes
    )  # noqa


def _create_empty_tensor(ms_type):
    init_func = Zero()
    init_func.__enable_zero_dim__ = True
    init_tensor = ms.Tensor(shape=(0,), dtype=ms_type, init=init_func)
    init_tensor.init_data()

    return init_tensor


def make_tensor_with_pad(
    x: List[List[T]],
    pad: T,
    dtype: torch.dtype,
    *,
    max_len: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Make a padded tensor from 2D inputs.

    The padding is applied to the end of each inner list until it reaches
    `max_len`.
    """
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    pin_memory = False

    if padded_x.size == 0:
        tensor = _create_empty_tensor(dtype)
    else:
        tensor = torch.from_numpy(padded_x)
    if pin_memory:
        tensor = tensor.pin_memory()

    return tensor


def async_tensor_h2d(
    data: list,
    dtype: torch.dtype,
    target_device: Union[str, torch.device],
    pin_memory: bool,
) -> torch.Tensor:
    """Asynchronously create a tensor and copy it from host to device."""
    if not data:
        t = _create_empty_tensor(dtype)
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=pin_memory, device="CPU")
    return t


STR_DTYPE_TO_TENSOR_DTYPE = {
    "half": torch.half,
    "float16": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

STR_DTYPE_TO_MS_DTYPE = {
    "half": mstype.float16,
    "float16": mstype.float16,
    "bfloat16": mstype.bfloat16,
    "float": mstype.float32,
    "fp8": mstype.uint8,
    "fp8_e4m3": mstype.uint8,
    "fp8_e5m2": mstype.uint8,
}


def get_dtype_size(dtype: torch.dtype) -> int:
    """Get the size of the data type in bytes."""
    if isinstance(dtype, str):
        dtype = STR_DTYPE_TO_TENSOR_DTYPE[dtype]
    return torch.tensor([1], dtype=dtype).itemsize


def ascend_device_count_stateless() -> List[str]:
    visible_device_str = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)
    if visible_device_str:
        try:
            res = visible_device_str.split(",")
        except Exception as e:
            logger.error('Cannot parse "ASCEND_RT_VISIBLE_DEVICES" for: %s!' % str(e))
            raise ValueError(
                'Error argument(%s) of environ "ASCEND_RT_VISIBLE_DEVICES"!'
                % visible_device_str
            )

        return len(res)

    import re
    import subprocess

    output = subprocess.check_output(["npu-smi", "info"], encoding="utf-8")
    res = re.findall(
        r"\|\s+\d+\s+\w+\s+\|\s+(\w+)\s+\|\s+(?:[0-9\.]+|-)\s+[0-9\.]+\s+\d+\s+\/\s+\d+\s+\|",
        output,
    )

    avl_devices = []
    for i, stat in enumerate(res):
        if stat != "OK":
            logger.warning("Device %d is not ok, status is %s!" % (i, stat))
        else:
            avl_devices.append(str(i))
    visible_device_str = ",".join(avl_devices)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_device_str
    logger.info('Set environ "ASCEND_RT_VISIBLE_DEVICES" as %s' % visible_device_str)

    return len(avl_devices)


def ascend_is_initialized():
    # Just return true for check.
    return True


def is_mindformers_model_backend():
    return (
        os.getenv("vLLM_MODEL_BACKEND")
        and os.environ["vLLM_MODEL_BACKEND"] == "MindFormers"
    )


def check_ready():
    from mindspore import set_context

    # Common environment variables of predict.
    set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    if is_mindformers_model_backend():
        logger.info("Run with Mindformers backend!")
        necessary_envs = ("vLLM_MODEL_MEMORY_USE_GB", "MINDFORMERS_MODEL_CONFIG")
        lost_envs = [env_item for env_item in necessary_envs if not os.getenv(env_item)]

        if lost_envs:
            raise RuntimeError(
                'For "MindFormers" model backend, environments %s should be set!'
                % str(lost_envs)
            )

        mindformers_default_env = {
            "MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST": "FlashAttentionScore,PagedAttention",
        }
        env_setup(mindformers_default_env)

        set_context(mode=0, device_target="Ascend", max_call_depth=10000)
    else:
        logger.info("Run with native model backend!")


def cal_block_num(cache_config, model_config, parallel_config):
    from vllm.worker.cache_engine import CacheEngine

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    _, total_gpu_memory = torch.cuda.mem_get_info()
    memory_can_use = total_gpu_memory * cache_config.gpu_memory_utilization

    model_use_memory_b = int(os.getenv("vLLM_MODEL_MEMORY_USE_GB")) * 1024 * 1024 * 1024
    available_cache_memory = memory_can_use - model_use_memory_b
    cache_block_size = CacheEngine.get_cache_block_size(
        cache_config, model_config, parallel_config
    )
    num_gpu_blocks = int(available_cache_memory // cache_block_size)
    return num_gpu_blocks
