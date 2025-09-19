# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.8.3/vllm/config.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import socket
import threading
import time
from collections import Counter
from typing import Any, Literal, Optional, Union

import msgspec
import torch
import vllm.envs as envs
from pydantic.dataclasses import dataclass
from transformers import PretrainedConfig
from vllm.compilation.inductor_pass import CallableInductorPass, InductorPass
from vllm.config import (_STR_DTYPE_TO_TORCH_DTYPE, CacheConfig,
                         CompilationConfig, CompilationLevel, VllmConfig,
                         _find_dtype, _resolve_auto_dtype, get_attr_docs)
from vllm.logger import init_logger
from vllm.utils import random_uuid

from vllm_mindspore.utils import is_310p

logger = init_logger(__name__)


def _verify_quantization(self) -> None:
    # Do not verify now.
    return


def vllm_config_post_init(self):
    """Verify configs are valid & consistent with each other."""
    if self.model_config is not None:
        self.model_config.verify_async_output_proc(self.parallel_config,
                                                   self.speculative_config,
                                                   self.device_config)
        self.model_config.verify_with_parallel_config(self.parallel_config)

    if self.cache_config is not None:
        self.cache_config.verify_with_parallel_config(self.parallel_config)

    if self.lora_config:
        self.lora_config.verify_with_cache_config(self.cache_config)
        self.lora_config.verify_with_model_config(self.model_config)
        self.lora_config.verify_lora_support()
    if self.prompt_adapter_config:
        self.prompt_adapter_config.verify_with_model_config(self.model_config)

    if self.quant_config is None and \
        self.model_config is not None and self.load_config is not None:
        self.quant_config = VllmConfig._get_quantization_config(
            self.model_config, self.load_config)

    from vllm.platforms import current_platform
    if self.scheduler_config is not None and \
        self.model_config is not None and \
        self.scheduler_config.chunked_prefill_enabled and \
        self.model_config.dtype == torch.float32 and \
        current_platform.get_device_capability() == (7, 5):
        logger.warning_once(
            "Turing devices tensor cores do not support float32 matmul. "
            "To workaround this limitation, vLLM will set 'ieee' input "
            "precision for chunked prefill triton kernels.")

    if self.compilation_config is None:
        self.compilation_config = CompilationConfig()
    if envs.VLLM_USE_V1 and self.model_config is not None and \
        not self.model_config.enforce_eager:
        # NOTE(woosuk): Currently, we use inductor because the piecewise
        # CUDA graphs do not work properly with the custom CUDA kernels.
        # FIXME(woosuk): Disable inductor to reduce the compilation time
        # and avoid any potential issues with the inductor.
        self.compilation_config.custom_ops = ["none"]
        self.compilation_config.use_cudagraph = True
        self.compilation_config.use_inductor = True
        self.compilation_config.cudagraph_num_of_warmups = 1
        self.compilation_config.pass_config.enable_fusion = False
        self.compilation_config.pass_config.enable_noop = False
        # When level is set to CompilationLevel.PIECEWISE, vllm will use cuda
        # graph, which means the model inputs will be padded to cuda graph
        # acceptable size, but it is not for mindspore.
        # So here set to CompilationLevel.DYNAMO_AS_IS.
        self.compilation_config.level = CompilationLevel.DYNAMO_AS_IS
        # Set a small compile_sizes for warmup. '20' is not in
        # 'cudagraph_capture_sizes'. So the warmup can be run.
        self.compilation_config.compile_sizes = [20]

    self._set_cudagraph_sizes()

    if self.cache_config is not None and \
        self.cache_config.cpu_offload_gb > 0 and \
        self.compilation_config.level != CompilationLevel.NO_COMPILATION:
        logger.warning("CPU offload is not supported with `torch.compile` yet."
                       " Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    if self.lora_config is not None and self.compilation_config.level !=\
            CompilationLevel.NO_COMPILATION:
        logger.warning("LoRA is not supported with `torch.compile` yet. "
                       "Disabling `torch.compile`.")
        self.compilation_config.level = CompilationLevel.NO_COMPILATION

    current_platform.check_and_update_config(self)

    if self.model_config and self.model_config.use_mla:
        logger.info(
            "For MindSpore, MLA supports chunked prefill and prefix cache, "
            "so keep them enable.")

    if not self.instance_id:
        self.instance_id = random_uuid()[:5]


def _verify_args(self) -> None:
    if (self.max_num_batched_tokens < self.max_model_len
            and not self.chunked_prefill_enabled):
        logger.warning(
            "max_num_batched_tokens (%d) is smaller than max_model_len (%d). "
            "This effectively limits the maximum sequence length to "
            "max_num_batched_tokens and makes vLLM reject longer "
            "sequences. Please increase max_num_batched_tokens or "
            "decrease max_model_len.", self.max_num_batched_tokens,
            self.max_model_len)

    if self.max_num_batched_tokens < self.max_num_seqs:
        raise ValueError(
            f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
            "be greater than or equal to max_num_seqs "
            f"({self.max_num_seqs}).")

    if self.num_lookahead_slots < 0:
        raise ValueError(
            "num_lookahead_slots "
            f"({self.num_lookahead_slots}) must be greater than or "
            "equal to 0.")

    if self.num_scheduler_steps < 1:
        raise ValueError(
            "num_scheduler_steps "
            f"({self.num_scheduler_steps}) must be greater than or "
            "equal to 1.")

    if self.max_num_partial_prefills < 1:
        raise ValueError(
            f"max_num_partial_prefills ({self.max_num_partial_prefills}) "
            "must be greater than or equal to 1.")
    elif self.max_num_partial_prefills > 1:
        if not self.chunked_prefill_enabled:
            raise ValueError("Chunked prefill must be enabled to set "
                             "max_num_partial_prefills > 1.")

        if self.long_prefill_token_threshold > self.max_model_len:
            raise ValueError(
                "long_prefill_token_threshold "
                f"({self.long_prefill_token_threshold}) cannot be greater "
                f"than the max_model_len ({self.max_model_len}).")

    if (self.max_long_partial_prefills
            < 1) or (self.max_long_partial_prefills
                     > self.max_num_partial_prefills):
        raise ValueError(
            f"max_long_partial_prefills ({self.max_long_partial_prefills}) "
            "must be greater than or equal to 1 and less than or equal to "
            f"max_num_partial_prefills ({self.max_num_partial_prefills}).")


def model_post_init(self, __context) -> None:

    count_none = self.custom_ops.count("none")
    count_all = self.custom_ops.count("all")
    assert count_none + count_all <= 1, "Can only specify 'none' or 'all'"

    if self.splitting_ops is None:
        self.splitting_ops = []

    for k, v in self.inductor_passes.items():
        if not isinstance(v, str):
            assert callable(v), (
                f"pass {k} should be callable or a qualified name")
            self.inductor_compile_config[k] = v if isinstance(
                v, InductorPass) else CallableInductorPass(v)
            continue

        # resolve function from qualified name
        names = v.split(".")
        module = ".".join(names[:-1])
        func_name = names[-1]
        func = __import__(module).__dict__[func_name]
        self.inductor_compile_config[k] = func if isinstance(
            func, InductorPass) else CallableInductorPass(func)

    self.enabled_custom_ops = Counter()
    self.disabled_custom_ops = Counter()
    self.traced_files = set()
    self.static_forward_context = {}
    self.compilation_time = 0.0


def _get_and_verify_dtype(
    model_id: str,
    config: PretrainedConfig,
    dtype: Union[str, torch.dtype],
    *,
    is_pooling_model: bool,
    revision: Optional[str] = None,
) -> torch.dtype:
    config_dtype = _find_dtype(model_id, config, revision=revision)
    model_type = config.model_type

    if isinstance(dtype, str):
        dtype = dtype.lower()
        if dtype == "auto":
            # Set default dtype from model config
            torch_dtype = _resolve_auto_dtype(
                model_type,
                config_dtype,
                is_pooling_model=is_pooling_model,
            )
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unknown dtype: {dtype!r}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]
    elif isinstance(dtype, torch.dtype):
        torch_dtype = dtype
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    if torch_dtype == torch.bfloat16 and is_310p():
        torch_dtype = torch.float16

    if torch_dtype != config_dtype:
        if torch_dtype == torch.float32:
            # Upcasting to float32 is allowed.
            logger.info("Upcasting %s to %s.", config_dtype, torch_dtype)
        elif config_dtype == torch.float32:
            # Downcasting from float32 to float16 or bfloat16 is allowed.
            logger.info("Downcasting %s to %s.", config_dtype, torch_dtype)
        else:
            # Casting between float16 and bfloat16 is allowed with a warning.
            logger.warning("Casting %s to %s.", config_dtype, torch_dtype)

    if torch_dtype in _STR_DTYPE_TO_TORCH_DTYPE:
        torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[torch_dtype]

    return torch_dtype


class SocketProcessGroup:

    def __init__(self, master_ip: str, master_port: int, rank: int,
                 world_size: int):
        self.master_ip = master_ip
        self.master_port = master_port
        self.rank = rank
        self.world_size = world_size
        self.sockets: list[socket.socket] = []
        self.max_retries = 100
        self.retry_interval = 2
        self.conn_thread: Optional[threading.Thread] = None

        if self.rank == 0:
            # Master node: create a server socket
            self.server_socket = socket.socket(socket.AF_INET,
                                               socket.SOCK_STREAM)
            self.server_socket.bind((self.master_ip, self.master_port))
            self.server_socket.listen(self.world_size - 1)
            logger.info("Master node listening on %s:%d", self.master_ip,
                        self.master_port)
        else:
            # Worker node: connect to the master
            self.client_socket = socket.socket(socket.AF_INET,
                                               socket.SOCK_STREAM)
            retries = 0
            while retries < self.max_retries:
                try:
                    self.client_socket.connect(
                        (self.master_ip, self.master_port))
                    logger.info("Worker %d connected to master at %s:%d",
                                self.rank, self.master_ip, self.master_port)
                    break
                except ConnectionRefusedError:
                    retries += 1
                    logger.warning(
                        "Worker %d failed to connect to master. "
                        "Retrying in %d seconds... (%d/%d)", self.rank,
                        self.retry_interval, retries, self.max_retries)
                    time.sleep(self.retry_interval)
            else:
                raise ConnectionError(
                    f"Worker {self.rank} could not connect to master at "
                    f"{self.master_ip}:{self.master_port} after "
                    f"{self.max_retries} retries.")

    def accept_connections(self):
        for _ in range(self.world_size - 1):
            conn, addr = self.server_socket.accept()
            print(f"Accepted connection from {addr}")
            self.sockets.append(conn)

    def initialize_group(self):
        if self.rank == 0:
            # Master node: accept connections from workers
            self.conn_thread = threading.Thread(target=self.accept_connections,
                                                daemon=True)
            self.conn_thread.start()
        else:
            # Worker node: no additional setup needed
            self.conn_thread = None

    def close(self):
        if self.rank == 0:
            # Master node: close all worker connections
            for conn in self.sockets:
                conn.close()
            self.server_socket.close()
        else:
            # Worker node: close connection to master
            self.client_socket.close()


def stateless_init_dp_group(self) -> SocketProcessGroup:
    """
    Initialize a stateless data parallel process group using sockets.
    """
    dp_group = SocketProcessGroup(self.data_parallel_master_ip,
                                  self.get_next_dp_init_port(),
                                  self.data_parallel_rank,
                                  self.data_parallel_size)
    dp_group.initialize_group()
    return dp_group


def has_unfinished_dp(dp_group: SocketProcessGroup,
                      has_unfinished: bool) -> bool:
    """
    Check if any process in the group has unfinished tasks.
    """
    if dp_group.rank == 0:
        # Master node: collect results from workers
        assert dp_group.conn_thread is not None
        # Wait for all dp engine connectioned.
        dp_group.conn_thread.join()
        results = [has_unfinished]
        for conn in dp_group.sockets:
            data = conn.recv(1024)
            worker_result = msgspec.msgpack.decode(data)
            results.append(worker_result)

        # Perform OR operation (any True means unfinished)
        aggregated_result = any(results)

        # Broadcast the result back to workers
        for conn in dp_group.sockets:
            conn.send(msgspec.msgpack.encode(aggregated_result))

        return aggregated_result
    else:
        # Worker node: send result to master
        dp_group.client_socket.send(msgspec.msgpack.encode(has_unfinished))

        # Receive aggregated result from master
        data = dp_group.client_socket.recv(1024)
        aggregated_result = msgspec.msgpack.decode(data)
        return aggregated_result


def stateless_destroy_socket_process_group(
        dp_group: "SocketProcessGroup") -> None:
    """
    Destroy the socket-based data parallel process group.
    This function closes all sockets and cleans up resources.
    """
    if dp_group:
        dp_group.close()
        logger.info("Socket process group for rank %d destroyed.",
                    dp_group.rank)


#The location of the native vllm:
#https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/config.py#1447
#Compared with it, "int8" was added.
CacheDType = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "int8"]

# the vllm.config.get_attr_docs function can only obtain the' docs' of
# the current class, but not the 'docs' of the member variables of the
# parent class.
get_current_class_attr_docs = get_attr_docs


def get_current_and_parent_class_attr_docs(cls: type[Any]) -> dict[str, str]:
    '''
    Due to the get_attr_docs function limitation of native vllm, we have added
    this function to obtain the docs of the attributes of the parent class and
    the child class.

    The reason for adding this function is that we are going to add the
    _CacheConfig class, which inherits from the CacheConfig class, but we don't
    want to add all the member variables of the parent class.
    '''
    parent_docs = {}
    for base in cls.__bases__:
        if base is object:
            continue
        # get docs of parent class
        parent_docs.update(get_current_class_attr_docs(base))
    # get docs of current class
    current_docs = get_current_class_attr_docs(cls)
    # Merge the parent class and the current class
    parent_docs.update(current_docs)
    return parent_docs


@dataclass
class _CacheConfig(CacheConfig):
    '''
    Configuration for the KV cache.
    This _CacheConfig has one modification compared to the original vllm
    CacheConfig class:
    1.the data type of the cache_dtype member variable: support int8.(new
    CacheDType Literal in vllm_mindspore/config.py#L405.)

    The reason for not using the patch method:
    if the data type of the cache_dtype member variable is modified by
    changing the __annotations__ of the original vllm CacheConfig class,
    an error will be reported during the initialization of the
    CacheConfig class at
    https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/engine/arg_utils.py#L1060
    for the int8 verification.
    The possible reason is that modifying __annotations__ is ineffective
    for pydantic; a new class must be created.

    The reason for Remove the @config decorator from the current class:
    The get_attr_docs method in
    https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/config.py#L191
    fails to obtain the member 'docs' from the parent class,
    resulting in an error during subsequent validation when use the
    @config decorator.
    '''

    cache_dtype: CacheDType = "auto"
    """Data type for kv cache storage. If "auto", will use model data type.
    CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. ROCm (AMD GPU) supports
    fp8 (=fp8_e4m3)."""
