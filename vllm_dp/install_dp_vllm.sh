#!/bin/bash
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
# ============================================================================

# This bash is to apply njhill's Multi-node server solution 
# (https://github.com/vllm-project/vllm/pull/15906, https://github.com/vllm-project/vllm/pull/15977)
# to vllm v0.8.3 and install it.

script_dir=$(cd "$(dirname $0)"; pwd)
old_dir=$(pwd)

vllm_tag="v0.8.3"

vllm_source_dir="${script_dir}/vllm-${vllm_tag}"

if [ -d "${vllm_source_dir}" ]; then
    echo "The ${vllm_source_dir} already exists, install maybe done! If not, please remove and rename it first."
    exit 1
fi

git clone https://github.com/vllm-project/vllm.git -b ${vllm_tag} --depth 1 ${vllm_source_dir}
cd ${vllm_source_dir}

git apply "${script_dir}/dp_scale_out.patch"

export VLLM_TARGET_DEVICE=empty
pip install .

cd ${old_dir}