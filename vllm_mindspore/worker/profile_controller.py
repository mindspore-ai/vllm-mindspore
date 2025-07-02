#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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


import os
import json
import sys
import subprocess
import tarfile
import shutil
from types import SimpleNamespace

import mindspore as ms

# host profiling modules
from mindspore._c_expression import _framework_profiler_enable_mi
from mindspore._c_expression import _framework_profiler_disable_mi
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end
from mindspore._c_expression import _framework_profiler_clear

# device profiling utils
from mindspore import Profiler
from mindspore.profiler import ProfilerLevel,  ProfilerActivity
from mindspore.profiler.common.profiler_context import ProfilerContext

# vllm modules
import vllm.envs as envs
from vllm.logger import init_logger
from vllm.entrypoints.openai.api_server import router as vllm_router
from vllm.entrypoints.openai.api_server import engine_client

from fastapi import Request
from fastapi.responses import Response, JSONResponse, FileResponse, HTMLResponse

from vllm_mindspore.dashboard_utils import get_dashboard_html


VLLM_DEFAULT_PROFILE_ENV_NAME = "VLLM_TORCH_PROFILING_DIR"
VLLM_MS_PROFILE_CONFIG_PATH_ENV_NAME = "VLLM_MS_PROFILE_CONFIG_PATH"

# default vllm-mindspore profile config is based on the vllm backend start dir
# the content example is like follow
# {
#     "enable_profile": true,
#     "profile_config": {
#         "profile_type": "device",
#         "start_iteration": 50,
#         "sample_iteration": 10,
#         "profile_output_path": "./graph",
#         "online_ananlyse": true,
#         "profiler_level": "Level1",
#         "with_stack": true,
#         "activities": ["CPU", "NPU"]
#     }
# }
DEFAULT_VLLM_MS_CONFIG_FILE_PATH = "./vllm_ms_profile.config"

vllm_logger = init_logger(__name__)

def shell_analyse(path: str) -> None:
    subprocess.run(
        [sys.executable, "-c", f'from mindspore import Profiler; Profiler.offline_analyse("{path}")'],
        shell=False, check=True
    )
    return


# Pure vLLM MindSpore Profile Config class
class ProfileControllerConfig:
    def __init__(self):
        # start_iteration: iterations to run before real profile
        self.start_iteration = 50
        # sample_iteration: iteration num to profile
        self.sample_iteration = 10
        # profile_type: device or host profile, advice use device
        self.profile_type = "device"
        # profile_output_path: output path of profiling
        self.profile_output_path = "./graph"
        # online_analyse: if online analyse profile data
        self.online_ananlyse = True
        # profiler_level: device profiler level, valid value: Level0/Level1/Level2
        # Note: the string must the same with valid value
        self.profiler_level = ProfilerLevel.Level1
        # with_stack: if profile python stack data
        self.with_stack = True
        # activities: the profile active,  it is a List with "CPU", "NPU", "GPU"
        # advice always use ["CPU", "NPU"] on Ascend platform
        self.activities = [ProfilerActivity.CPU, ProfilerActivity.NPU]

    def to_dict(self):
        out_dict = {}

        for (key, value) in self.__dict__.items():
            if hasattr(value, "to_dict"):
                out_dict[key] = value.to_dict()
            elif isinstance(value, ProfilerLevel):
                out_dict[key] = value.value
            elif key == "activities":
                # ctivities is a list of ProfilerActivity Enum, deal it single case
                out_list = []
                for elem in value:
                    out_list.append(str(elem.value))
                out_dict[key] = out_list
            else:
                out_dict[key] = value
        return out_dict


default_profile_config = ProfileControllerConfig()
# this avariable is because origin vLLM profiler is controlled by the output path
# in vllm-mindspore, the output path is package files dir
profile_results_path = os.getenv(VLLM_DEFAULT_PROFILE_ENV_NAME, "./profile_results")

# Control profile class
class ProfileController:
    def __init__(self, config: ProfileControllerConfig = default_profile_config):
        self.name = "vllm mindspore profile controller"
        self.is_profiling = False
        self.config = config
        self.iteration = 0
        self.profiler = None

    
    # start profile controll period
    def start(self, config: ProfileControllerConfig = None) -> None:
        if self.is_profiling:
            # already in profiling state, skip
            vllm_logger.warning(f"vllm-mindspore is already in profiling state, try start later")
            return
        
        self.is_profiling = True
        if config is not None:
            vllm_logger.info(f"start profile with new config: {config.to_dict()}")
            self.config = config
        
        self.iteration = 0

    
    # host profile check point function
    def _host_profile_point(self) -> None:
        if self.iteration == self.config.start_iteration:
            # start host profile
            if os.environ.get("MS_ENABLE_RUNTIME_PROFILER", "") != "1":
                vllm_logger.warning(f"env MS_ENABLE_RUNTIME_PROFILER is not set, host profile cannot work")
            vllm_logger.info(f"start host profile at iteration {self.iteration}")
            # set the host output path
            ms.set_context(save_graphs_path=self.config.profile_output_path)
            _framework_profiler_enable_mi()
            _framework_profiler_step_start()


        if self.iteration == self.config.start_iteration + self.config.sample_iteration:
            # end host profile
            vllm_logger.info(f"end host profile at iteration {self.iteration}")
            _framework_profiler_step_end()
            _framework_profiler_clear()
            _framework_profiler_disable_mi()
            self.is_profiling = False

        return
    
    # device profile check point function
    def _device_profile_point(self) -> None:
        if self.iteration == self.config.start_iteration:
            # start device profile
            self.profiler = Profiler(profiler_level=self.config.profiler_level, 
                                     activities=self.config.activities, 
                                     with_stack=self.config.with_stack, 
                                     output_path=self.config.profile_output_path)
        

        if self.iteration == self.config.start_iteration + self.config.sample_iteration:
            # end device profile
            vllm_logger.info(f"end device profile at iteration {self.iteration}")
            self.profiler.stop()
            self.is_profiling = False

        return   
    

    # if the controller is in profiling state
    def is_profiling(self) -> bool:
        return self.is_profiling


    # exposed profile control check point function
    def check_profile_point(self):
        if not self.is_profiling:
            # controller is not in profilig state, return
            return
        
        if self.config.profile_type == "host":
            self._host_profile_point()
        elif self.config.profile_type == "device":
            self._device_profile_point()
        else:
            vllm_logger.warning(f"Invalid profiling type {self.config.profile_type}, please check profile config")
            self.is_profiling = False
            self.iteration = 0

        self.iteration += 1

    
    # stop profile controll period
    def stop(self):
        if self.config.profile_type == "device":
            if self.is_profiling:
                # the profile is not finish, stop it
                if self.profiler:
                    self.profiler.stop()
                self.is_profiling = False

            if self.profiler and self.config.online_ananlyse:
                # enable online analyse, call analyse
                try:
                    self.profiler.analyse()
                except Exception as e:
                    vllm_logger.warning(f"the online analyse catch exception {e}, try offline analyse.")
                    profile_output_path = ProfilerContext().ascend_ms_dir
                    shell_analyse(profile_output_path)
            self.profiler = None


vllm_mindspore_profile_controller = ProfileController()


# class for file config for profile controller
# this is used for changing profile config when vLLM is already running
# because vLLM do not provide set config pai for profiling, 
# so vllm-mindspore reuse the api, and set the config from specified file path
# the file path is set by a env VLLM_MS_PROFILE_CONFIG_PATH when vLLM server setup
# if the config file is not exist, the profile controller will use default config
class ProfileFileControlerConfig(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self):
        out_dict = {}

        for (key, value) in self.__dict__.items():
            if hasattr(value, "to_dict"):
                out_dict[key] = value.to_dict()
            else:
                out_dict[key] = value
        
        return out_dict


default_profile_file_controller_config = ProfileFileControlerConfig()
# enable_profile: if the profile is anable, if the config set False, call start will not start profile
default_profile_file_controller_config.enable_profile = True
# profile_config: the profile config
default_profile_file_controller_config.profile_config = ProfileControllerConfig()


# the Profiler class for vLLM, it will take the start and stop api from vLLM to control profile
class AdapterControlProfiler:
    def __init__(self, config_path: str):
        self.config_path = config_path
    
    def get_config(self):
        if not os.path.exists(self.config_path):
            # config file path is not exist, return default profile config
            vllm_logger.info(f"profile config path is not exist, use default config")
            return default_profile_file_controller_config
        
        with open(self.config_path, "r") as config_file:
            config_json = config_file.read()
            try:
                config = json.loads(config_json, object_hook=lambda d: ProfileFileControlerConfig(**d))
            except Exception as e:
                vllm_logger.warning(f"invalid profile config file, return default config")
                return default_profile_file_controller_config
            
        return config
    
    def start(self):
        # only start call will trigger read config file
        config = self.get_config()
        if not config.enable_profile:
            # config file disable profile, print warning to tell user
            vllm_logger.warning(f"the config file is disable the profile, please check it again")

        vllm_mindspore_profile_controller.start(config.profile_config)

    
    def stop(self):
        vllm_mindspore_profile_controller.stop()
        
        # package the profile result
        current_profile_output_path = ProfilerContext().ascend_ms_dir
        vllm_logger.info(f"packaging the profile dir: {current_profile_output_path}")
        
        profile_dir_name = os.path.basename(current_profile_output_path)
        package_profile_file_path = f"{profile_results_path}/{profile_dir_name}.tar.gz"
        
        with tarfile.open(package_profile_file_path, "w:gz") as tar:
            tar.add(current_profile_output_path, arcname=os.path.basename(current_profile_output_path))


# the profile controller init function, if the vLLM is not enable profile, this init function will provide the api
def init_vllm_mindspore_profile_controller() -> None:
    # in vllm-mindspore, the profile api is always provided for easy to use
    # so we do not need restart vllm if we want to profile
    # if the VLLM_TORCH_PROFILING_DIR env is set, the vLLLM will set the api
    if not envs.VLLM_TORCH_PROFILER_DIR:
        @vllm_router.post("/start_profile")
        async def start_profile(raw_request: Request):
            vllm_logger.info("Starting profiler...")
            await engine_client(raw_request).start_profile()
            vllm_logger.info("Profiler started.")
            return Response(status_code=200)
        
        @vllm_router.post("/stop_profile")
        async def stop_profile(raw_request: Request):
            vllm_logger.info("Stop profiler...")
            await engine_client(raw_request).stop_profile()
            vllm_logger.info("Profiler stopped.")
            return Response(status_code=200)
        
    # get the profile config path
    # the reason for this api is like above, we do not want to modify vLLM source code to provide profile ability
    @vllm_router.get("/get_profile_config_info")
    async def get_profile_config_path(raw_request: Request):
        profile_config_path = os.getenv(VLLM_MS_PROFILE_CONFIG_PATH_ENV_NAME, DEFAULT_VLLM_MS_CONFIG_FILE_PATH)
        ret = {"vllm_ms_profile_config_path": profile_config_path,
               "vllm_ms_profile_config_example": default_profile_file_controller_config.to_dict()}
        return JSONResponse(ret)
    
    @vllm_router.get("/get_profile_result_files")
    async def get_profile_result_files(raw_request: Request):
        profile_result_file_list = os.listdir(profile_results_path)
        
        ret = {
            "vllm_ms_profile_files": profile_result_file_list
        }
        return JSONResponse(ret)
    
    @vllm_router.get("/get_profile_data/{file_name}")
    async def get_profile_data(file_name: str):
        profile_file_path = f"{profile_results_path}/{file_name}"
        vllm_logger.info(f"packaging the profile dir: {profile_file_path}")
        return FileResponse(profile_file_path, filename=file_name)
    
    @vllm_router.get("/profile_dashboard")
    async def get_profile_data(raw_request: Request):
        vllm_logger.info(f"raw_request: {raw_request}")
        dashboard_html_str = get_dashboard_html()
        return HTMLResponse(dashboard_html_str)
    
    return

# wrapper vLLM worker init functions
# these functions instead the vLLM worker init to init profiler modules
def wrapper_worker_init(func) -> None:
    def new_func(*args, **kwargs) -> None:
        # Profiler initialization during worker init triggers device setup,
        # causing init device to fail due to duplicate configuration.
        # To fix this, temporarily unset VLLM_TORCH_PROFILING_DIR before vLLM worker init,
        # restore it afterward, then initialize profiler properlly after worker init_device completes
        profile_output_path = os.getenv(VLLM_DEFAULT_PROFILE_ENV_NAME, "")
        if profile_output_path:
            del os.environ[VLLM_DEFAULT_PROFILE_ENV_NAME]

        func(*args, **kwargs)

        if profile_output_path:
            os.environ[VLLM_DEFAULT_PROFILE_ENV_NAME] = profile_output_path
    return new_func

def wrapper_worker_init_device(func) -> None:
    def new_func(*args, **kwargs):
        func(*args, **kwargs)

        # The actual profiler initialization is performed after the worker.init_device() method,
        # based on the VLLM_TORCH_PROFILING_DIR environment variable.
        worker = args[0]
        profile_config_path = os.getenv(VLLM_MS_PROFILE_CONFIG_PATH_ENV_NAME, DEFAULT_VLLM_MS_CONFIG_FILE_PATH)

        # reset profile results dir
        if os.path.exists(profile_results_path):
            shutil.rmtree(profile_results_path, ignore_errors=True)
        os.makedirs(profile_results_path, exist_ok=True)
        
        worker.profiler = AdapterControlProfiler(profile_config_path)
    return new_func

