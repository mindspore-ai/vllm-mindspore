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

"""Base class for code generator."""

import os
import re
from jinja2 import Environment, FileSystemLoader


class CodeGenerator:
    """
    Base code generator using Jinja2 templates to generate code.
    For template design details, refer to https://jinja.palletsprojects.com/en/stable/templates/.
    """

    def __init__(self, template_dir: list[str], output_dir: str):
        """
        Args:
            template_dir: List of directories containing Jinja2 templates.
            output_dir: Directory to output the generated code.
        """
        self.template_dir = template_dir
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        self.gen_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )

        # Common filters
        self.gen_env.filters['to_capitalize_case'] = self._to_capitalize_case
        self.gen_env.filters['to_snake_case'] = self._to_snake_case
        self.gen_env.filters['to_upper_case'] = self._to_upper_case

    def generate(self, **kwargs) -> None:
        """Generate code from data to output directory."""
        raise NotImplementedError("Subclasses must implement this method.")

    def _to_capitalize_case(self, text: str) -> str:
        """Convert text to CapitalizeCase."""
        if not text:
            return text
        words = text.replace('_', ' ').replace('.', ' ').split()
        return ''.join(word.capitalize() for word in words)

    def _to_snake_case(self, text: str) -> str:
        """Convert text to snake_case."""
        if not text:
            return text
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    def _to_upper_case(self, text: str) -> str:
        """Convert text to UPPER_CASE."""
        if not text:
            return text
        return text.upper()
