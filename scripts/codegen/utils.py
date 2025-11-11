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

"""Utility functions for codegen."""

import subprocess
import shutil
import os
from pathlib import Path
import textwrap


def _find_clang_format():
    """Find clang-format executable in the system."""
    clang_format_names = ['clang-format', 'clang-format-15', 'clang-format-14',
                          'clang-format-13', 'clang-format-12', 'clang-format-11']

    for name in clang_format_names:
        if shutil.which(name):
            return shutil.which(name)
    return None

def format_directory(directory):
    """Format all C++ files in a directory using clang-format."""
    directory = Path(directory)

    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return

    # Find clang-format
    clang_format_path = _find_clang_format()
    if clang_format_path is None:
        print("Warning: clang-format not found. Please install clang-format.")
        return

    # Find C++ files
    cpp_files = []
    for pattern in ["*.h", "*.cc"]:
        cpp_files.extend(directory.rglob(pattern))

    if not cpp_files:
        print(f"No C++ files found in {directory}")
        return

    print(f"Using clang-format: {clang_format_path}")

    for file_path in cpp_files:
        try:
            # Run clang-format
            subprocess.run([clang_format_path, "-i", str(file_path)],
                           capture_output=True, text=True, check=True)
            print(f"Formatted: {file_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error formatting {file_path}: {e}")

    print("Apply clang-format for directory", directory)

def text_wrap(text, width=80):
    """Wrap text to a given width."""
    return textwrap.fill(text, width, subsequent_indent='    ')

def save_file(content: str, output_dir: str, file_name: str):
    """Save content to a file."""
    with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
        f.write(content)
