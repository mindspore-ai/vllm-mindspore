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

readonly IMAGE_TAG="vllm_ms_$(date +%Y%m%d)"
ARCH=$([ "$(uname -m)" = "x86_64" ] && echo "x86_64" || echo "aarch64")

log() {
    echo "========= $*"
}

main() {
    log "Starting Docker image build process"
    
    log "Step 1: Building base environment for $ARCH"
    docker image inspect base_env >/dev/null 2>&1 || docker build --build-arg TARGETARCH=$ARCH -t base_env -f Dockerfile .
    
    log "Step 2: Building final image with install script"
    
    # Get mindformers commit ID from submodule
    local mindformers_commit=$(git submodule status tests/mindformers | awk '{print $1}' | sed 's/^-//')
    log "Using mindformers commit: $mindformers_commit"
    
    cat > Dockerfile.tmp << EOF
FROM base_env

COPY install_depend_pkgs.sh /workspace/
COPY .jenkins/test/config/dependent_packages.yaml /workspace/.jenkins/test/config/

ARG MINDFORMERS_COMMIT=$mindformers_commit
RUN chmod +x /workspace/install_depend_pkgs.sh && \\
    cd /workspace && \\
    MINDFORMERS_COMMIT=\$MINDFORMERS_COMMIT ./install_depend_pkgs.sh

ENV PYTHONPATH="/workspace/mindformers/:\$PYTHONPATH"
WORKDIR /workspace

CMD ["bash"]
EOF
    
    docker build -f Dockerfile.tmp -t "$IMAGE_TAG" . || { 
        echo "Failed to build final image"
        rm -f Dockerfile.tmp
        exit 1
    }
    rm -f Dockerfile.tmp
    
    log "Build completed: $IMAGE_TAG"
}

main "$@"
