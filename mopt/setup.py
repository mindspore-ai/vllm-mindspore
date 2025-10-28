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

        third_party_dir = str(Path(__file__).parent.parent / "build" / "third_party")
        llvm_dir = os.environ.get(
            "LLVM_DIR", third_party_dir + "/install/llvm/lib/cmake/llvm"
        )
        mlir_dir = os.environ.get(
            "MLIR_DIR", third_party_dir + "/install/llvm/lib/cmake/mlir"
        )
        llvm_external_lit = os.environ.get(
            "LLVM_EXTERNAL_LIT", third_party_dir + "/build/llvm/bin/llvm-lit"
        )

        cmake_args = [
            f"-DMLIR_DIR={mlir_dir}",
            f"-DLLVM_DIR={llvm_dir}",
            f"-DLLVM_EXTERNAL_LIT={llvm_external_lit}",
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

        # Copy the generated package
        python_package_dir = Path(self.build_temp) / "python_packages" / "mopt"
        target_dir = Path(self.build_lib) / "mopt"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(python_package_dir, target_dir)


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
