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
from setuptools import Extension
import subprocess
import warnings


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)
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


def write_commit_id():
    ret_code = os.system("git rev-parse --abbrev-ref HEAD > ./vllm_mindspore/.commit_id "
                         "&& git log --abbrev-commit -1 >> ./vllm_mindspore/.commit_id")
    if ret_code != 0:
        sys.stdout.write("Warning: Can not get commit id information. Please make sure git is available.")
        os.system("echo 'git is not available while building.' > ./vllm_mindspore/.commit_id")


version = (Path("vllm_mindspore") / "version.txt").read_text()

def _get_ascend_home_path():
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

def _get_ascend_env_path():
    env_script_path = os.path.realpath(os.path.join(_get_ascend_home_path(), "..", "set_env.sh"))
    if not os.path.exists(env_script_path):
        raise ValueError(f"The file '{env_script_path}' is not found, "
                            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly.")
    return env_script_path

class CustomBuildExt(build_ext):
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    def build_extension(self, ext):
        if ext.name == "vllm_mindspore.npu_ops":
            self.build_npu_ops(ext)
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")

    def build_npu_ops(self, ext):
        # "vllm_mindspore.npu_ops" --> "npu_ops"
        ext_name = ext.name.split('.')[-1]
        so_name = ext_name + ".so"
        logger.info(f"Building {so_name} ...")
        OPS_DIR = os.path.join(ROOT_DIR, "vllm_mindspore", "ops")
        BUILD_OPS_DIR = os.path.join(ROOT_DIR, "build", "ops")
        os.makedirs(BUILD_OPS_DIR, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_script_path = _get_ascend_env_path()
        build_extension_dir = os.path.join(BUILD_OPS_DIR, "kernel_meta", ext_name)
        # Combine all cmake commands into one string
        cmake_cmd = (
            f"source {env_script_path} && "
            f"cmake -S {OPS_DIR} -B {BUILD_OPS_DIR}"
            f"  -DCMAKE_BUILD_TYPE=Release"
            f"  -DCMAKE_INSTALL_PREFIX={os.path.join(BUILD_OPS_DIR, 'install')}"
            f"  -DBUILD_EXTENSION_DIR={build_extension_dir}"
            f"  -DMS_EXTENSION_NAME={ext_name}"
            f"  -DASCEND_CANN_PACKAGE_PATH={ascend_home_path} && "
            f"cmake --build {BUILD_OPS_DIR} -j --verbose"
        )

        try:
            # Run the combined cmake command
            logger.info(f"Running combined CMake commands:\n{cmake_cmd}")
            result = subprocess.run(cmake_cmd, cwd=self.ROOT_DIR, text=True, shell=True, capture_output=True)
            if result.returncode != 0:
                logger.info("CMake commands failed:")
                logger.info(result.stdout)  # Print standard output
                logger.info(result.stderr)  # Print error output
                raise RuntimeError(f"Combined CMake commands failed with exit code {result.returncode}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {so_name}: {e}")

        # Copy the generated .so file to the target directory
        src_so_path = os.path.join(build_extension_dir, so_name)
        dst_so_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(dst_so_path), exist_ok=True)
        if os.path.exists(dst_so_path):
            os.remove(dst_so_path)
        shutil.copy(src_so_path, dst_so_path)
        logger.info(f"Copied {so_name} to {dst_so_path}")


write_commit_id()

package_data = {
    "": [
        "*.so",
        "lib/*.so",
        ".commit_id"
    ]
}

def _get_ext_modules():
    ext_modules = []
    if os.path.exists(_get_ascend_home_path()):
        # sources are specified in CMakeLists.txt
        ext_modules.append(Extension("vllm_mindspore.npu_ops", sources=[]))
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
    entry_points={
        "console_scripts": [
            "vllm-mindspore=vllm_mindspore.scripts:main",
        ],
    },
)
