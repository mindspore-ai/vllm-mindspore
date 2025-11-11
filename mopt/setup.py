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

"""Setup script for MRT MLIR Python package."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class CMakeBuild(build_ext):
    """Custom build command to compile C++ extensions."""

    def run(self):
        """Build C++ extensions using CMake."""
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        llvm_dir = os.environ.get("LLVM_DIR")
        mlir_dir = os.environ.get("MLIR_DIR")

        cmake_args = [
            f"-DMLIR_DIR={mlir_dir}",
            f"-DLLVM_DIR={llvm_dir}",
        ]

        subprocess.check_call(
            [
                "cmake",
                "-S",
                str(Path(__file__).parent),
                "-B",
                self.build_temp,
            ]
            + cmake_args
        )
        build_jobs = os.environ.get("BUILD_JOBS", "8")
        subprocess.check_call(["cmake", "--build", self.build_temp, "-j", build_jobs])

        # Copy the generated mopt package
        python_package_dir = Path(self.build_temp) / "python_packages" / "mopt"
        target_dir = Path(self.build_lib) / "mopt"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(python_package_dir, target_dir)

        # Package torch_mlir as a top-level package
        # (parallel to mopt in site-packages)
        # torch_mlir is generated during compilation, package from build path
        build_dir = Path(__file__).parent.parent / "build"
        torch_mlir_path = (build_dir / "third_party" / "build" / "llvm" /
                          "tools" / "torch-mlir" / "python_packages" /
                          "torch_mlir" / "torch_mlir")

        if torch_mlir_path.is_dir():
            torch_mlir_dst = Path(self.build_lib) / "torch_mlir"
            if torch_mlir_dst.exists():
                shutil.rmtree(torch_mlir_dst)
            shutil.copytree(torch_mlir_path, torch_mlir_dst)
            print(f"Packaged torch_mlir from {torch_mlir_path}")
        else:
            print(f"Warning: torch_mlir not found at {torch_mlir_path}")


class BuildPyWithExt(build_py):
    def run(self):
        super().run()
        self.run_command("build_ext")


setup(
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPyWithExt,
    }
)
