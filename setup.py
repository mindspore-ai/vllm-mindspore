# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 Huawei Technologies Co., Ltd.
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
"""setup package."""

import importlib.util
import logging
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(
        module_name, path)  # type: ignore[union-attr]
    module = importlib.util.module_from_spec(
        spec)  # type: ignore[union-attr, arg-type]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
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


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> list[str]:
        with open(get_path(filename)) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--") or "http" in line:
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


def write_commit_id():
    commit_info = ""
    try:
        commit_info += subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode("utf-8")
        commit_info += subprocess.check_output(
            ["git", "log", "--abbrev-commit", "-1"]).decode("utf-8")
    except subprocess.CalledProcessError:
        logger.warning("Can't get commit id information. "
                       "Please make sure git is available.")
        commit_info = "git is not available while building."

    with open("./vllm_mindspore/.commit_id", "w") as f:
        f.write(commit_info)


version = (Path("vllm_mindspore") / "version.txt").read_text()


def _get_ascend_home_path():
    return os.environ.get("ASCEND_HOME_PATH",
                          "/usr/local/Ascend/ascend-toolkit/latest")


def _get_ascend_env_path():
    env_script_path = os.path.realpath(
        os.path.join(_get_ascend_home_path(), "..", "set_env.sh"))
    if not os.path.exists(env_script_path):
        raise ValueError(
            "The file '{}' is not found, please make sure environment "
            "variable 'ASCEND_HOME_PATH' is set correctly.".format(
                env_script_path))
    return env_script_path


class CustomBuildExt(build_ext):
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    def build_extension(self, ext):
        if ext.name == "vllm_mindspore._C_ops":
            self.build_c_ops(ext)
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")

    def build_c_ops(self, ext):
        # "vllm_mindspore._C_ops" --> "_C_ops"
        ext_name = ext.name.split('.')[-1]
        so_name = ext_name + ".so"
        logger.info("Building %s ...", so_name)
        OPS_DIR = os.path.join(ROOT_DIR, "csrc")
        BUILD_OPS_DIR = os.path.join(ROOT_DIR, "build", "csrc_ops")
        if os.path.exists(BUILD_OPS_DIR):
            shutil.rmtree(BUILD_OPS_DIR)
        os.makedirs(BUILD_OPS_DIR, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_script_path = _get_ascend_env_path()
        build_extension_dir = os.path.join(BUILD_OPS_DIR, "kernel_meta",
                                           ext_name)
        # Combine all cmake commands into one string
        cmake_cmd = (
            f"source {shlex.quote(env_script_path)} && "
            f"cmake -S {OPS_DIR} -B {BUILD_OPS_DIR}"
            f"  -DCMAKE_BUILD_TYPE=Release"
            f"  -DCMAKE_INSTALL_PREFIX={os.path.join(BUILD_OPS_DIR, 'install')}"
            f"  -DBUILD_EXTENSION_DIR={build_extension_dir}"
            f"  -DMS_EXTENSION_NAME={ext_name}"
            f"  -DASCEND_CANN_PACKAGE_PATH={shlex.quote(ascend_home_path)} && "
            f"cmake --build {BUILD_OPS_DIR} -j --verbose")

        # Run the combined cmake command
        logger.info("Running commands:\n%s", cmake_cmd)
        build_log_file = os.path.join(BUILD_OPS_DIR, "build_log.txt")
        with open(build_log_file, "w") as log_file:
            result = subprocess.run(["bash", "-c", cmake_cmd],
                                    cwd=self.ROOT_DIR,
                                    text=True,
                                    stdout=log_file,
                                    stderr=log_file)
        if result.returncode != 0:
            logger.error("Command failed: '%s' exited with code %s", cmake_cmd,
                         result.returncode)
            raise RuntimeError(
                "Failed to build {}, check the build log for details: {}".
                format(ext_name, build_log_file))

        # Copy the generated .so file to the target directory
        src_so_path = os.path.join(build_extension_dir, so_name)
        dst_so_path = self.get_ext_fullpath(ext.name)
        os.makedirs(os.path.dirname(dst_so_path), exist_ok=True)
        if os.path.exists(dst_so_path):
            os.remove(dst_so_path)
        shutil.copy(src_so_path, dst_so_path)
        logger.info("Build %s succeeded.", dst_so_path)


write_commit_id()

package_data = {"": ["*.so", "lib/*.so", ".commit_id"]}


def _get_ext_modules():
    ext_modules = []
    if os.path.exists(_get_ascend_home_path()):
        # sources are specified in CMakeLists.txt
        ext_modules.append(Extension("vllm_mindspore._C_ops", sources=[]))
    return ext_modules


setup(
    name="vllm-mindspore",
    version=version,
    author="MindSpore Team",
    license="Apache 2.0",
    description=("A high-throughput and memory-efficient inference and "
                 "serving engine for LLMs"),
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
