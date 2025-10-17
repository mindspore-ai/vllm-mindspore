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
"""Main entry point for monkey patching vllm."""

# isort:skip_file

import os

external_mode = bool(int(os.getenv("USE_MS_BACKEND", '0')))
if not external_mode:
    import msadapter

    from vllm_mindspore.ray_patch import patch_ray
    patch_ray()

    import vllm_mindspore.apply_patch import vllm_patch_enable
    vllm_patch_enable()
