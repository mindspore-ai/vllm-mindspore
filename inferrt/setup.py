"""Setup script for InferRT Python package."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


def get_git_commit_id():
    """Get the current git commit ID."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


class CMakeBuild(build_ext):
    """Custom build command to compile C++ extensions."""

    def _get_cmake_args_from_env(self):
        """Extract CMake arguments from environment variables."""
        cmake_args = []

        # Environment variable to CMake argument mappings
        env_mappings = {
            "DEBUG": None,  # Direct passthrough (no -D prefix)
            "DEBUG_LOG_OUT": None,  # Direct passthrough (no -D prefix)
            "BUILD_TESTS": "BUILD_TESTS=on",
            "ENABLE_ASCEND": "ENABLE_ASCEND=on",
            "ENABLE_CPU": "ENABLE_CPU=on",
            "ENABLE_TORCH_FRONT": "ENABLE_TORCH_FRONT=on",
            "ENABLE_GITEE": "ENABLE_GITEE=on",
        }

        for env_var, cmake_arg in env_mappings.items():
            env_value = os.environ.get(env_var)
            if not env_value:
                continue

            if cmake_arg is None:
                # Split space-separated flags and append each individually
                for item in env_value.lstrip().split():
                    cmake_args.append(item)
            elif env_value == "1":
                # Enable CMake option when environment variable is "1"
                cmake_args.append(f"-D{cmake_arg}")

        return cmake_args

    def run(self):
        """Build C++ extensions using CMake."""
        install_dir = Path(self.build_lib) / "ms_inferrt"
        if install_dir.exists():
            shutil.rmtree(install_dir)
        install_dir.mkdir(parents=True, exist_ok=True)

        # Set up CMake arguments
        cmake_args = [
            f"-DCMAKE_INSTALL_PREFIX={install_dir}",
            f"-DPython_EXECUTABLE={sys.executable}",
        ]

        # Add environment-based arguments
        cmake_args.extend(self._get_cmake_args_from_env())
        print("cmake_args:", cmake_args)

        # Configure with CMake
        if os.environ.get("INC_BUILD", "0") != "1":
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

        # Build with CMake
        build_jobs = os.environ.get("BUILD_JOBS", "8")
        subprocess.check_call(["cmake", "--build", self.build_temp, "-j", build_jobs])

        # Install to build directory
        subprocess.check_call(["cmake", "--install", self.build_temp])

        # Get the commit ID and write it to a file
        git_commit_id = get_git_commit_id()
        commit_id_file = install_dir / ".commit-id"
        commit_id_file.write_text(git_commit_id)


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
