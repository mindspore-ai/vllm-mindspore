#!/usr/bin/env bash

set -euo pipefail

readonly IMAGE_TAG="vllm_ms_$(date +%Y%m%d)"

log() {
    echo "========= $*"
}

main() {
    log "Starting Docker image build process"
    
    # Step 1: Build base environment image
    log "Step 1: Building base environment"
    docker image inspect base_env >/dev/null 2>&1 || docker build -t base_env -f Dockerfile .
    
    # Step 2: Build final image using install script
    log "Step 2: Building final image with install script"
    
    cat > Dockerfile.tmp << EOF
FROM base_env

# Copy installation script and config
COPY install_depend_pkgs.sh /workspace/
COPY .jenkins/test/config/dependent_packages.yaml /workspace/.jenkins/test/config/

# Make script executable and run installation
RUN chmod +x /workspace/install_depend_pkgs.sh && \\
    cd /workspace && \\
    ./install_depend_pkgs.sh

# Set environment and working directory
ENV PYTHONPATH="/workspace/mindformers/:\$PYTHONPATH"
ENV vLLM_MODEL_BACKEND=MindFormers
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