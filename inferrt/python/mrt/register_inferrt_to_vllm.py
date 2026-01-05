# Copyright 2026 Huawei Technologies Co., Ltd
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
"""
One-click patch to add InferrtAdaptor support to vllm
Usage: Add 'from mrt import register_inferrt_to_vllm' at the beginning of your code
"""

from typing import Optional, Any, Callable
from torch import fx
from mrt.torch import backend
from vllm.compilation.compiler_interface import CompilerInterface
from vllm.compilation.counter import compilation_counter
from vllm.compilation import backends

def register_to_vllm():
    """Main patch function to add InferrtAdaptor support with one click"""

    # 1. Add InferrtAdaptor class
    class InferrtAdaptor(CompilerInterface):
        """
        Adaptor for integrating InferRT compiler backend with VLLM.
        
        This class provides an interface for compiling torch.fx graph modules
        using the InferRT backend within the VLLM compilation framework.
        """
        name = "inferrt"

        def compile(
            self,
            graph: fx.GraphModule,
            example_inputs: list[Any],
            compiler_config: dict[str, Any],
            runtime_shape: Optional[int] = None,
            key: Optional[str] = None,
        ) -> tuple[Optional[Callable], Optional[Any]]:
            """
            Compile a torch.fx graph module using InferRT backend.

            Returns:
                A tuple containing the compiled callable and optional metadata.
            """
            _ = compiler_config, runtime_shape, key
            compilation_counter.num_eager_compiles += 1
            return backend(graph, example_inputs), None

    # 2. Patch the make_compiler function
    original_make_compiler = backends.make_compiler

    def patched_make_compiler(compilation_config):
        if compilation_config.backend == "inferrt":
            return InferrtAdaptor()
        return original_make_compiler(compilation_config)

    backends.make_compiler = patched_make_compiler

    print("vllm successfully patched with InferrtAdaptor support")
    return True

# Auto-execute patch on import
try:
    register_to_vllm()
except Exception as e:
    print(f"Failed to register inferrt to vllm: {e}")
    print("  vllm may not be installed or there's a version incompatibility.")

# Export functions
__all__ = ['register_to_vllm']
