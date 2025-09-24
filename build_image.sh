#!/usr/bin/env bash
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

set -euo pipefail
TARGET="910b"
ARCH=$(uname -m)

build_usage() {
    echo "Usage: $0 [-a 910b|310p]"
    exit 1
}

while getopts "a:h" opt; do
    case $opt in
        a)
            if [[ "$OPTARG" == "910b" || "$OPTARG" == "310p" ]]; then
                TARGET="$OPTARG"
            else
                echo "Error: -a must be '910b' or '310p', got '$OPTARG'" >&2
                exit 1
            fi
            ;;
        h)
            build_usage
            ;;
        \?|:)
            build_usage
            ;;
    esac
done

shift $((OPTIND - 1))
if [ $# -gt 0 ]; then
    echo "Error: invalid input $@"
    build_usage
fi

readonly IMAGE_TAG="vllm_ms_$(date +%Y%m%d)"

log() {
    echo "===== Build image for ${TARGET} ==== $*"
}

CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-toolkit_8.1.RC1_linux-${ARCH}.run
CANN_KERNELS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-kernels-${TARGET}_8.1.RC1_linux-${ARCH}.run
CANN_NNRT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.1.RC1/Ascend-cann-nnrt_8.1.RC1_linux-${ARCH}.run

add_verbose() {
    if docker build -h 2>&1 | grep -q "\-\-progress"; then
        build_args="--no-cache --progress=plain"
    else
        build_args="--no-cache"
    fi
}

build_args=""

get_run_opts() {
    while getopts ":vh" opt; do
        case $opt in
            v)
                add_verbose
                ;;
            h)
                echo "Usage: bash build_image.sh [-v]"
                echo "    verbose: print docker build logs in details"
                exit 0
                ;;
        esac
    done
}

main() {
    log "Starting Docker image build process"
    
    log "Step 1: Building base environment"

    docker image inspect base_env >/dev/null 2>&1 || docker build -t base_env -f Dockerfile . ${build_args}
    
    log "Step 2: Building final image with install script"
    
    # Get mindformers commit ID from submodule
    local mindformers_commit=$(git submodule status tests/mindformers | awk '{print $1}' | sed 's/^-//')
    log "Using mindformers commit: $mindformers_commit"
    
    cat > Dockerfile.tmp << EOF
FROM base_env

RUN set -ex && \\
    cd /root && \\
    wget --header="Referer: https://www.hiascend.com/" ${CANN_TOOLKIT_URL} -O Ascend-cann-toolkit.run --no-check-certificate && \\
    wget --header="Referer: https://www.hiascend.com/" ${CANN_KERNELS_URL} -O Ascend-cann-kernels-${TARGET}.run --no-check-certificate && \\
    wget --header="Referer: https://www.hiascend.com/" ${CANN_NNRT_URL} -O Ascend-cann-nnrt.run --no-check-certificate

RUN set -ex && \\
    cd /root && \\
    chmod a+x *.run && \\
    bash /root/Ascend-cann-toolkit.run --install -q && \\
    bash /root/Ascend-cann-kernels-${TARGET}.run --install -q && \\
    bash Ascend-cann-nnrt.run --install -q && \\
    rm /root/*.run

COPY install_depend_pkgs.sh /workspace/
COPY .jenkins/test/config/dependent_packages.yaml /workspace/.jenkins/test/config/
ADD ./ /workspace/vllm_mindspore

ARG MINDFORMERS_COMMIT=$mindformers_commit
RUN chmod +x /workspace/vllm_mindspore/install_depend_pkgs.sh && \\
    cd /workspace/vllm_mindspore && \\
    MINDFORMERS_COMMIT=\$MINDFORMERS_COMMIT AUTO_BUILD=1 ./install_depend_pkgs.sh

RUN cd /workspace/vllm_mindspore && \\
    pip install .

ENV PYTHONPATH="/workspace/mindformers/:\$PYTHONPATH"
WORKDIR /workspace

CMD ["bash"]
EOF
    
    docker build -f Dockerfile.tmp -t "$IMAGE_TAG" . ${build_args}|| { 
        echo "Failed to build final image"
        rm -f Dockerfile.tmp
        exit 1
    }
    rm -f Dockerfile.tmp
    
    log "Build completed: $IMAGE_TAG"
}

get_run_opts "$@"

main "$@"
