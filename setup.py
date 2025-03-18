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

BUILD_DIR = os.path.join(ROOT_DIR, "build")
KERNELS_SO_DIR = os.path.join(BUILD_DIR, "lib64")
KERNELS_SO_PATH = os.path.join(KERNELS_SO_DIR, "libascendc_kernels_npu.so")
NPU_OPS_SO_DIR = os.path.join(ROOT_DIR, "kernel_meta", "npu_ops")
NPU_OPS_SO_PATH = os.path.join(NPU_OPS_SO_DIR, "npu_ops.so")


class CustomBuildExt(build_ext):
    def run(self):
        # Step 1: Build libascendc_kernels_npu.so using the shell script
        self.build_kernels()
        # Step 2: Compile custom ops into npu_ops.so using CustomOpBuilder
        self.build_npu_ops()
        super().run()

    def build_kernels(self):
        print("Building libascendc_kernels_npu.so ...")
        soc_version = os.getenv("SOC_VERSION", None)
        build_script = os.path.join(ROOT_DIR, "vllm_mindspore", "ops", "scripts", "build_kernels.sh")
        if not os.path.exists(build_script):
            raise FileNotFoundError(f"Build script {build_script} not found.")
        cmd = ["bash", build_script, "-p", BUILD_DIR]
        if soc_version:
            cmd.extend(["-v", soc_version])
        result = subprocess.run(cmd, cwd=ROOT_DIR, check=True)
        if result.returncode != 0:
            raise RuntimeError("Failed to build libascendc_kernels_npu")
        print(f"libascendc_kernels_npu.so built successfully and placed in {KERNELS_SO_PATH}.")

    def build_npu_ops(self):
        print("Building npu_ops.so ...")
        import mindspore as ms
        try:
            ops_dir = os.path.join(ROOT_DIR, "vllm_mindspore", "ops")
            src = os.path.join(ops_dir, "adapt", "advance_step_flashattn.cpp")
            ms.ops.CustomOpBuilder("npu_ops", src, backend="Ascend",
                            cflags=f"-I{ops_dir}",
                            ldflags=f"-L{KERNELS_SO_DIR} -lascendc_kernels_npu -Wl,-rpath,'$$ORIGIN/lib'").load()
        except ImportError:
            pass
        print(f"npu_ops.so built successfully and placed in {NPU_OPS_SO_DIR}.")


class CustomInstall(install):
    def run(self):
        # Run the standard install process
        super().run()

        # Ensure `npu_ops.so` is included in the installed package
        self.include_npu_ops()

    def include_npu_ops(self):
        print("Including npu_ops.so and libascendc_kernels_npu.so in the installed package...")
        package_path = os.path.join(self.install_lib, "vllm_mindspore")
        lib_path = os.path.join(package_path, "lib")
        os.makedirs(lib_path, exist_ok=True)
        self.copy_file(NPU_OPS_SO_PATH, package_path)
        print("copy", NPU_OPS_SO_PATH, "to", package_path)
        self.copy_file(KERNELS_SO_PATH, lib_path)
        print("copy", KERNELS_SO_PATH, "to", lib_path)

cmdclass = {}
ext_modules = []
if os.getenv("vLLM_USE_NPU_ADV_STEP_FLASH_OP", "off") == "on":
    cmdclass["build_ext"] = CustomBuildExt
    cmdclass["install"] = CustomInstall
    # Dummy extension to trigger build_ext
    ext_modules.append(Extension("dummy", sources=[]))

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
    include_package_data=True,
    cmdclass=cmdclass,
    ext_modules=ext_modules,
)
