#!/bin/bash
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

# 使用方法: ./update_all.sh [date]

DATE=$1

# 如果 DATE 不为空，就加上 --date 参数
if [ -n "$DATE" ]; then
    DATE_ARG="--date=$DATE"
else
    DATE_ARG=""
fi

python update_package.py mindspore --branch r2.7.1 --no-cleanup $DATE_ARG
python update_package.py mindformers --branch r1.7.0 --no-cleanup $DATE_ARG
python update_package.py golden-stick --branch r1.3.0 --no-cleanup --whl-keyword mindspore_gs $DATE_ARG
python update_package.py msadapter --branch r0.3.0 --no-cleanup $DATE_ARG
python update_package.py vllm-mindspore --no-cleanup --whl-keyword vllm_mindspore $DATE_ARG


