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
"""setup package."""

import importlib.util
import logging
import os
import sys
import shutil
from typing import List
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools import Extension
import subprocess


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


if not sys.platform.startswith("linux"):
    logger.warning(
        "vllm_mindspore only supports Linux platform."
        "Building on %s, "
        "so vllm_mindspore may not be able to run correctly",
        sys.platform,
    )


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            elif "http" in line:
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


version = (Path("vllm_mindspore") / "version.txt").read_text()

def _get_ascend_home_path():
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

class CustomBuildExt(build_ext):
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
    ASCENDC_OPS_DIR = os.path.join(ROOT_DIR, "vllm_mindspore", "ops", "ascendc")

    def build_extension(self, ext):
        if ext.name == "ascendc_kernels_npu":
            self.build_ascendc_kernels()
        elif ext.name == "npu_ops":
            self.build_npu_ops(ext)
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")

    def build_ascendc_kernels(self):
        kernel_so_name = "libascendc_kernels_npu.so"
        print(f"Building {kernel_so_name}...")
        tmp_build_dir = os.path.join(self.ASCENDC_OPS_DIR, "build")
        if os.path.exists(tmp_build_dir):
            print(f"Removing existing build directory: {tmp_build_dir}")
            shutil.rmtree(tmp_build_dir)
        os.makedirs(tmp_build_dir, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_script_path = os.path.join(ascend_home_path, "bin", "setenv.bash")
        if not os.path.exists(env_script_path):
            raise RuntimeError(f"The file '{env_script_path}' is not found, "
                               "please make sure env variable 'ASCEND_HOME_PATH' is set correctly.")
        # Combine all cmake commands into one string
        cmake_cmd = (
            f"source {env_script_path} && "
            f"cmake -S {self.ASCENDC_OPS_DIR} -B {tmp_build_dir} "
            f"-DRUN_MODE=npu -DCMAKE_BUILD_TYPE=Debug "
            f"-DCMAKE_INSTALL_PREFIX={os.path.join(tmp_build_dir, 'install')} "
            f"-DASCEND_CANN_PACKAGE_PATH={ascend_home_path} && "
            f"cmake --build {tmp_build_dir} -j --verbose && "
            f"cmake --install {tmp_build_dir}"
        )

        try:
            # Run the combined cmake command
            print("Running combined CMake commands:")
            result = subprocess.run(cmake_cmd, cwd=self.ROOT_DIR, text=True, shell=True, capture_output=True)
            if result.returncode != 0:
                print("CMake commands failed:")
                print(result.stdout)  # Print standard output
                print(result.stderr)  # Print error output
                raise RuntimeError(f"Combined CMake commands failed with exit code {result.returncode}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {kernel_so_name}: {e}")

        # Move the generated .so file to the target directory
        src_so_path = os.path.join(tmp_build_dir, "lib", kernel_so_name)
        lib_dir = os.path.join(self.ROOT_DIR, self.build_lib, "vllm_mindspore", "lib")
        dst_so_path = os.path.join(lib_dir, kernel_so_name)
        os.makedirs(lib_dir, exist_ok=True)
        if os.path.exists(dst_so_path):
            os.remove(dst_so_path)
        shutil.move(src_so_path, dst_so_path)
        print(f"Moved {kernel_so_name} to {lib_dir}.")
        # Remove the build directory after building kernels.so
        shutil.rmtree(tmp_build_dir)

    def build_npu_ops(self, ext):
        print("Building npu_ops.so ...")
        try:
            import mindspore as ms
        except ImportError:
            print("Mindspore is not found, skip building npu_ops.so")
            return
        try:
            src = [os.path.join(self.ASCENDC_OPS_DIR, s) for s in ext.sources]
            build_lib_dir = os.path.join(self.ROOT_DIR, self.build_lib, "vllm_mindspore")
            ms.ops.CustomOpBuilder(
                "npu_ops",
                src,
                backend="Ascend",
                cflags=f"-I{self.ASCENDC_OPS_DIR}",
                ldflags=f"-L{os.path.join(build_lib_dir, 'lib')} -lascendc_kernels_npu -Wl,-rpath,'$$ORIGIN/lib'"
            ).load()
        except ImportError:
            pass
        # Move the generated .so file to the target directory
        kernel_meta_dir = os.path.join(self.ROOT_DIR, "kernel_meta")
        src_so_path = os.path.join(kernel_meta_dir, "npu_ops", "npu_ops.so")
        dst_so_path = os.path.join(build_lib_dir, "npu_ops.so")
        os.makedirs(build_lib_dir, exist_ok=True)
        if os.path.exists(dst_so_path):
            os.remove(dst_so_path)
        shutil.move(src_so_path, build_lib_dir)
        print(f"Moved npu_ops.so to {build_lib_dir}.")
        shutil.rmtree(kernel_meta_dir)

package_data = {
    "": [
        "*.so",
        "lib/*.so",
    ]
}

def _get_ext_modules():
    ext_modules = []
    # Currently, the CI environment does not support the compilation of custom operators.
    # As a temporary solution, this is controlled via an environment variable.
    # Once the CI environment adds support for custom operator compilation,
    # this should be updated to enable compilation by default.
    if os.getenv("vLLM_USE_NPU_ADV_STEP_FLASH_OP", "off") == "on":
        ext_modules.append(Extension("ascendc_kernels_npu", sources=[]))
        ext_modules.append(Extension("npu_ops", sources=[
            "adv_step_flash_adapter.cpp"
        ]))
    return ext_modules

setup(
    name="vllm-mindspore",
    version=version,
    author="MindSpore Team",
    license="Apache 2.0",
    description=(
        "A high-throughput and memory-efficient inference and "
        "serving engine for LLMs"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/vllm-mindspore",
    project_urls={
        "Homepage": "https://gitee.com/mindspore/vllm-mindspore",
        "Documentation": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=get_requirements(),
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=_get_ext_modules(),
    include_package_data=True,
    package_data=package_data,
)
