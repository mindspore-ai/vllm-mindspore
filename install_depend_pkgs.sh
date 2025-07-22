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

script_dir=$(cd "$(dirname $0)"; pwd)
yaml_file="$script_dir/.jenkins/test/config/dependent_packages.yaml"
work_dir="install_depend_pkgs"

if [ ! -f "$yaml_file" ]; then
    echo "$yaml_file does not exist."
    exit 1
fi

if [ ! -d "$work_dir" ]; then
    mkdir -p "$work_dir"
    echo "Created $work_dir directory."
else
    echo "$work_dir already exists. Removing existing whl packages."
    rm -f "$work_dir"/*.whl
fi

cd "$work_dir" || exit 1

get_yaml_value() {
    local file="$1"
    local key="$2"

    python3 -c "
import yaml
try:
    with open('$file', 'r') as f:
        data = yaml.safe_load(f)
        print(data.get('$key', ''))
except Exception as e:
    print(f'Error: {e}')
    exit(1)
"
}

echo "========= Installing vllm"
vllm_dir=vllm-v0.9.1
if [ ! -d "$vllm_dir" ]; then
    git clone https://github.com/vllm-project/vllm.git -b v0.9.1 "$vllm_dir"
    cd "$vllm_dir" ||  { echo "Failed to git clone vllm!"; exit 1; }
else
    echo "The $vllm_dir folder already exists and will not be re-downloaded."
    cd "$vllm_dir" || { echo "Failed to git clone vllm!"; exit 1; }
fi
pip uninstall msadapter -y
pip uninstall vllm -y
pip install -v -r requirements/cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=empty python setup.py install || { echo "Failed to install vllm"; exit 1; }
pip uninstall torch torch-npu torchvision torchaudio -y
cd ..


echo "========= Installing mindspore"
python_v="cp$(python3 --version 2>&1 | grep -oP 'Python \K\d+\.\d+' | tr -d .)"
mindspore_path=$(get_yaml_value "$yaml_file" "mindspore")
mindspore_name="mindspore-2.7.0-${python_v}-${python_v}-linux_$(arch).whl"
mindspore_pkg="${mindspore_path}unified/$(arch)/${mindspore_name}"

wget "$mindspore_pkg" --no-check-certificate || { echo "Failed to download mindspore"; exit 1; }
pip uninstall mindspore -y && pip install "$mindspore_name" || { echo "Failed to install mindspore"; exit 1; }


echo "========= Installing mindformers"
mf_dir=mindformers-br_infer_boom
if [ ! -d "$mf_dir" ]; then
    git clone https://gitee.com/mindspore/mindformers.git -b br_infer_boom "$mf_dir"
else
    echo "The $mf_dir folder already exists and will not be re-downloaded."
fi
if [ ! -d "$mf_dir" ]; then
    echo "Failed to git clone mindformers!"
    exit 1 
fi


echo "========= Installing mindspore golden-stick"
gs_dir=gs-develop
if [ ! -d "$gs_dir" ]; then
    git clone https://gitee.com/mindspore/golden-stick.git -b develop "$gs_dir"
else
    echo "The $gs_dir folder already exists and will not be re-downloaded."
fi
cd "$gs_dir" || { echo "Failed to git clone golden-stick!"; exit 1; }
pip uninstall mindspore-gs -y && pip install .|| { echo "Failed to install golden-stick"; exit 1; }
cd ..


echo "========= Installing msadapter"
msadapter_dir="MSAdapter"
if [ ! -d "$msadapter_dir" ]; then
    git clone https://git.openi.org.cn/OpenI/MSAdapter.git
else
    echo "The $msadapter_dir folder already exists and will not be re-downloaded."
fi
cd "$msadapter_dir" || { echo "Failed to git clone msadapter!"; exit 1; }
pip uninstall msadapter -y && pip install .  || { echo "Failed to install msadapter"; exit 1; }
cd ..

echo "========= All dependencies installed successfully!"
echo -e "[\033[0;34mnotice\033[0m] Please set the command: export PYTHONPATH=$(pwd)/$mf_dir/:\$PYTHONPATH"
