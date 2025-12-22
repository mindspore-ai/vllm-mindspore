# SPDX-License-Identifier: Apache-2.0

# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/v1/request.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2024-2025 The vLLM team.
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

import weakref
from functools import partial


def wrapper_request_init(fun):

    def new_fun(self, *args, **kwargs):
        fun(self, *args, **kwargs)

        block_hasher = kwargs.get("block_hasher")
        if block_hasher is None and args:
            block_hasher = args[-1]

        if block_hasher is not None and callable(block_hasher):
            # Override 'partial(block_hasher, self)' with
            # 'partial(block_hasher, proxy_self)' to avoid circular reference,
            # which may lead to memory leak.
            proxy_self = weakref.proxy(self)
            self.block_hashes = []  # Re-Initialize block_hashes
            self.get_hash_new_full_blocks = partial(block_hasher, proxy_self)
            self.block_hashes = self.get_hash_new_full_blocks()

    return new_fun
