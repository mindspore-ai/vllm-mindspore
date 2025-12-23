# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/executor/ray_distributed_executor.py
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/executor/ray_utils.py
# https://github.com/vllm-project/vllm/blob/v0.9.1/vllm/v1/utils.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
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
"""Adaption for ray executor."""

import os
from typing import Dict, List, Optional

from vllm.config import ParallelConfig, VllmConfig
from vllm.executor.ray_utils import RayWorkerWrapper
from vllm.logger import init_logger
from vllm.utils import get_ip
from vllm.v1.engine.core import DPEngineCoreActor
from vllm.v1.engine.utils import CoreEngineActorManager, EngineZmqAddresses
from vllm.v1.executor.abstract import Executor

logger = init_logger(__name__)

# Disable ray's automatic setting of ASCEND_RT_VISIBLE_DEVICES
# to avoid device recognition issues.
_RAY_NOSET_ASCEND_ENV = {
    "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES": "1"
}


class MsRayWorkerWrapper(RayWorkerWrapper):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.debug("Initialize RayWorkerWrapper with vllm-mindspore.")


class MsDPEngineCoreActor(DPEngineCoreActor):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        logger.debug("Initialize DPEngineCoreActor with vllm-mindspore.")


WORKER_SPECIFIC_ENV_VARS = {
    "VLLM_HOST_IP", "VLLM_HOST_PORT", "LOCAL_RANK", "ASCEND_RT_VISIBLE_DEVICES"
}


def initialize_ray_cluster(
    parallel_config: ParallelConfig,
    ray_address: Optional[str] = None,
):
    """Initialize the distributed cluster with Ray.

    it will connect to the Ray cluster and create a placement group
    for the workers, which includes the specification of the resources
    for each distributed worker.

    Args:
        parallel_config: The configurations for parallel execution.
        ray_address: The address of the Ray cluster. If None, uses
            the default Ray cluster address.
    """
    from vllm.executor.ray_utils import (_verify_bundles, _wait_until_pg_ready,
                                         assert_ray_available,
                                         available_resources_per_node, ray)

    assert_ray_available()
    from vllm.platforms import current_platform

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    elif current_platform.is_rocm() or current_platform.is_xpu():
        # Try to connect existing ray instance and create a new one if not found
        try:
            ray.init("auto")
        except ConnectionError:
            logger.warning(
                "No existing RAY instance detected. "
                "A new instance will be launched with current node resources.")
            ray.init(address=ray_address,
                     num_gpus=parallel_config.world_size,
                     runtime_env=parallel_config.ray_runtime_env)
    else:
        # vllm-mindspore: To prevent the issue of setting device failure caused
        # by ray override `ASCEND_RT_VISIBLE_DEVICES` override,
        # pass RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 as "env_vars"
        # parameter.
        from ray.runtime_env import RuntimeEnv

        if parallel_config.ray_runtime_env is None:
            init_runtime_env = RuntimeEnv(env_vars=_RAY_NOSET_ASCEND_ENV)
        else:
            init_runtime_env = parallel_config.ray_runtime_env
            init_runtime_env.set(
                "env_vars",
                init_runtime_env.env_vars() | _RAY_NOSET_ASCEND_ENV)
        ray.init(address=ray_address, runtime_env=init_runtime_env)
    device_str = current_platform.ray_device_key
    if not device_str:
        raise ValueError(
            f"current platform {current_platform.device_name} does not "
            "support ray.")

    # Create or get the placement group for worker processes
    if parallel_config.placement_group:
        current_placement_group = parallel_config.placement_group
    else:
        current_placement_group = ray.util.get_current_placement_group()

    if current_placement_group:
        logger.info("Using the existing placement group")

        # We are in a placement group
        bundles = current_placement_group.bundle_specs
        # Verify that we can use the placement group.
        device_bundles = 0
        for bundle in bundles:
            bundle_devices = bundle.get(device_str, 0)
            if bundle_devices > 1:
                raise ValueError(
                    "Placement group bundle cannot have more than 1 "
                    f"{device_str}.")
            if bundle_devices:
                device_bundles += 1
        if parallel_config.world_size > device_bundles:
            raise ValueError(
                f"The number of required {device_str}s exceeds the total "
                f"number of available {device_str}s in the placement group. "
                f"Required number of devices: {parallel_config.world_size}. "
                f"Total number of devices: {device_bundles}.")
    else:
        logger.info("No current placement group found. "
                    "Creating a new placement group.")
        num_devices_in_cluster = ray.cluster_resources().get(device_str, 0)
        # Log a warning message and delay resource allocation failure response.
        # Avoid immediate rejection to allow user-initiated placement group
        # created and wait cluster to be ready
        if parallel_config.world_size > num_devices_in_cluster:
            logger.warning(
                "The number of required %ss exceeds the total "
                "number of available %ss in the placement group.", device_str,
                device_str)
        # Create a new placement group
        placement_group_specs: List[Dict[str, float]] = ([{
            device_str: 1.0
        } for _ in range(parallel_config.world_size)])

        # vLLM engine is also a worker to execute model with an accelerator,
        # so it requires to have the device in a current node. Check if
        # the current node has at least one device.
        current_ip = get_ip()
        current_node_id = ray.get_runtime_context().get_node_id()
        current_node_resource = available_resources_per_node()[current_node_id]
        if current_node_resource.get(device_str, 0) < 1:
            raise ValueError(
                f"Current node has no {device_str} available. "
                f"{current_node_resource=}. vLLM engine cannot start without "
                f"{device_str}. Make sure you have at least 1 {device_str} "
                f"available in a node {current_node_id=} {current_ip=}.")
        # This way, at least bundle is required to be created in a current
        # node.
        placement_group_specs[0][f"node:{current_ip}"] = 0.001

        # By default, Ray packs resources as much as possible.
        current_placement_group = ray.util.placement_group(
            placement_group_specs, strategy="PACK")
        _wait_until_pg_ready(current_placement_group)

    assert current_placement_group is not None
    _verify_bundles(current_placement_group, parallel_config, device_str)
    # Set the placement group in the parallel config
    parallel_config.placement_group = current_placement_group


def core_engine_actor_manager_init(
    self,
    vllm_config: VllmConfig,
    addresses: EngineZmqAddresses,
    executor_class: type[Executor],
    log_stats: bool,
    placement_groups=None,
    local_dp_ranks=None,
):
    import copy

    import ray
    from ray.runtime_env import RuntimeEnv
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from vllm.platforms import current_platform
    from vllm.ray.ray_env import get_env_vars_to_copy
    from vllm.v1.engine.utils import get_device_indices

    self.local_engine_actors: list[ray.ActorHandle] = []
    self.remote_engine_actors: list[ray.ActorHandle] = []

    env_vars_list = get_env_vars_to_copy(destination="DPEngineCoreActor")
    self.env_vars_dict = {
        name: os.environ[name]
        for name in env_vars_list if name in os.environ
    }
    # vllm-mindspore: Add RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES for
    # remote call.
    runtime_env = RuntimeEnv(env_vars=self.env_vars_dict
                             | _RAY_NOSET_ASCEND_ENV)

    self.addresses = addresses
    self.executor_class = executor_class
    self.log_stats = log_stats
    dp_size = vllm_config.parallel_config.data_parallel_size
    local_engine_count = \
        vllm_config.parallel_config.data_parallel_size_local
    world_size = vllm_config.parallel_config.world_size

    if ray.is_initialized():
        logger.info("Ray is already initialized. Skipping Ray initialization.")
    else:
        # vllm-mindspore: To prevent the issue of setting device failure caused
        # by ray override `ASCEND_RT_VISIBLE_DEVICES` override,
        # pass RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1 as "env_vars"
        # parameter.
        ray.init(runtime_env={"env_vars": _RAY_NOSET_ASCEND_ENV})

    if placement_groups is not None:
        assert local_dp_ranks is not None, (
            "local_dp_ranks must be provided if "
            "placement_groups is provided")
        assert len(placement_groups) == len(local_dp_ranks), (
            "placement_groups and local_dp_ranks must "
            "have the same length")
        logger.info("Using provided placement groups")
        # TODO(rui): validate passed-in placement groups
        self.created_placement_groups = []
    else:
        placement_groups, local_dp_ranks = \
            CoreEngineActorManager.create_dp_placement_groups(vllm_config)
        self.created_placement_groups = placement_groups
    assert len(placement_groups) == dp_size, (
        "Number of placement groups must match data parallel size")

    self.placement_group_is_local = []
    refs = []
    for index, local_index, pg in zip(range(dp_size), local_dp_ranks,
                                      placement_groups):
        dp_vllm_config = copy.deepcopy(vllm_config)
        dp_vllm_config.parallel_config.placement_group = pg
        local_client = index < local_engine_count

        # Ray XPU known issue: dpctl initializes the GPU runtime early, so
        # setting device env vars in Ray actor's initialization method
        # will not affect device selection. See:
        # https://github.com/ray-project/ray/blob/master/python/ray/_private/accelerators/intel_gpu.py#L56 # noqa: E501
        if current_platform.is_xpu():
            device_evar = current_platform.device_control_env_var
            device_indices = get_device_indices(device_evar, local_index,
                                                world_size)
            actor_env_vars = self.env_vars_dict.copy()
            actor_env_vars[device_evar] = device_indices
            runtime_env = RuntimeEnv(env_vars=actor_env_vars)

        # vllm-mindspore: Use MsDPEngineCoreActor to enable vllm-mindspore
        # firstly.
        actor = ray.remote(MsDPEngineCoreActor).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=world_size,
            ),
            runtime_env=runtime_env).remote(vllm_config=dp_vllm_config,
                                            executor_class=executor_class,
                                            log_stats=log_stats,
                                            local_client=local_client,
                                            addresses=addresses,
                                            dp_rank=index,
                                            local_dp_rank=local_index)
        if local_client:
            self.local_engine_actors.append(actor)
        else:
            self.remote_engine_actors.append(actor)
        self.placement_group_is_local.append(local_client)
        refs.append(actor.wait_for_init.remote())

    ray.get(refs)
    self.run_refs = []
    for actor in self.local_engine_actors + self.remote_engine_actors:
        self.run_refs.append(actor.run.remote())
