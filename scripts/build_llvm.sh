#!/bin/bash

PROJECT_DIR="${PROJECT_DIR:-${PWD}}"

if [ "X${ENABLE_GITEE}" == "X1" ]; then
  LLVM_URL="https://gitee.com/mirrors/LLVM/repository/archive/d16b21b17d13ecd88a068bb803df43e53d3b04ba.zip"
  TORCHMLIR_URL="https://gitee.com/mirrors_llvm/torch-mlir/repository/archive/7e7af670802d99cacdaf26e6e37249d544e4896e.zip"
  STABLEHLO_URL="https://gitee.com/magicor/stablehlo/repository/archive/c28d55e91b4a5daaff18a33ce7e9bbd0f171256a.zip"
else
  LLVM_URL="https://github.com/llvm/llvm-project/archive/d16b21b17d13ecd88a068bb803df43e53d3b04ba.zip"
  TORCHMLIR_URL="https://github.com/llvm/torch-mlir/archive/7e7af670802d99cacdaf26e6e37249d544e4896e.zip"
  STABLEHLO_URL="https://github.com/openxla/stablehlo/archive/c28d55e91b4a5daaff18a33ce7e9bbd0f171256a.zip"
fi

TEMP_ARCHIVES_DIR="${PROJECT_DIR}/build/third_party/archives"
mkdir -p "${TEMP_ARCHIVES_DIR}"

download() {
  local url="$1"
  local out="$2"
  if [ -f "${out}" ] && [ -s "${out}" ]; then
    # TODO(dayschan): check hash value of archive file
    echo "Skip download, found existing archive: ${out}"
    return
  fi
  wget --no-check-certificate -O "${out}" "${url}"
}

extract_zip_to_dir() {
  local zip_file="$1"
  local dest_dir="$2"
  rm -rf "${dest_dir}"
  mkdir -p "${dest_dir}"
  local work_dir
  work_dir="$(mktemp -d "${TEMP_ARCHIVES_DIR}/unzip.XXXXXX")"
  unzip -q "${zip_file}" -d "${work_dir}"
  local top_dir
  top_dir="$(find "${work_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)"
  if [ -z "${top_dir}" ]; then
    echo "Error: failed to locate top dir in ${zip_file}" >&2
    exit 1
  fi
  shopt -s dotglob
  mv "${top_dir}/"* "${dest_dir}/"
  shopt -u dotglob
  rm -rf "${work_dir}"
}

LLVM_DIR="${PROJECT_DIR}/third_party/llvm-project"
TORCHMLIR_DIR="${PROJECT_DIR}/third_party/torch-mlir"
STABLEHLO_DIR="${TORCHMLIR_DIR}/externals/stablehlo"

LLVM_ZIP="${TEMP_ARCHIVES_DIR}/llvm-project-d16b21b17d13ecd88a068bb803df43e53d3b04ba.zip"
download "${LLVM_URL}" "${LLVM_ZIP}"
extract_zip_to_dir "${LLVM_ZIP}" "${LLVM_DIR}"

TORCHMLIR_ZIP="${TEMP_ARCHIVES_DIR}/torch-mlir-7e7af670802d99cacdaf26e6e37249d544e4896e.zip"
download "${TORCHMLIR_URL}" "${TORCHMLIR_ZIP}"
extract_zip_to_dir "${TORCHMLIR_ZIP}" "${TORCHMLIR_DIR}"

STABLEHLO_ZIP="${TEMP_ARCHIVES_DIR}/stablehlo-c28d55e91b4a5daaff18a33ce7e9bbd0f171256a.zip"
download "${STABLEHLO_URL}" "${STABLEHLO_ZIP}"
extract_zip_to_dir "${STABLEHLO_ZIP}" "${STABLEHLO_DIR}"

#------------ build llvm
LLVM_BUILD_DIR="${PROJECT_DIR}/build/third_party/build/llvm"
LLVM_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX:-${LLVM_BUILD_DIR}/install}"

cd "${PROJECT_DIR}"
echo "Configuring llvm..."
cmake -GNinja -B "${LLVM_BUILD_DIR}" \
    ${CCACHE_CMAKE_ARGS} \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON \
    -DMLIR_BUILD_MLIR_C_DYLIB=ON \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_ENABLE_PROJECTS="mlir" \
    -DLLVM_ENABLE_RTTI=ON \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_BUILD_UTILS=ON \
    -DLLVM_INSTALL_UTILS=ON \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PREFIX}" \
    -DCMAKE_BUILD_RPATH="${LLVM_BUILD_DIR}/lib" \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN/../lib" \
    third_party/llvm-project/llvm
echo "Configured llvm."

echo "Building llvm..."
cmake --build "${LLVM_BUILD_DIR}"
echo "Built llvm: ${LLVM_BUILD_DIR}"

echo "Installing llvm..."
cmake --install "${LLVM_BUILD_DIR}"
echo "Installed llvm: ${LLVM_INSTALL_PREFIX}"

#------------ build torch_mlir
TORCHMLIR_BUILD_DIR="${PROJECT_DIR}/build/third_party/build/torch_mlir"
TORCHMLIR_INSTALL_PREFIX="${TORCHMLIR_INSTALL_PREFIX:-${TORCHMLIR_BUILD_DIR}/install}"

echo "Configuring torch_mlir..."
cmake -GNinja -B "${TORCHMLIR_BUILD_DIR}" \
    ${CCACHE_CMAKE_ARGS} \
    -DMLIR_DIR="${LLVM_INSTALL_PREFIX}/lib/cmake/mlir" \
    -DLLVM_DIR="${LLVM_INSTALL_PREFIX}/lib/cmake/llvm" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DTORCH_MLIR_ENABLE_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="${TORCHMLIR_INSTALL_PREFIX}" \
    -DCMAKE_BUILD_RPATH="${TORCHMLIR_INSTALL_PREFIX}/lib" \
    -DTORCH_MLIR_INSTALL_USE_SYMLINKS=OFF \
    third_party/torch-mlir
echo "Configured torch_mlir."

echo "Building torch_mlir..."
cmake --build "${TORCHMLIR_BUILD_DIR}"
echo "Built torch_mlir: ${TORCHMLIR_BUILD_DIR}"

echo "Installing torch_mlir..."
cmake --install "${TORCHMLIR_BUILD_DIR}"
echo "Installed torch_mlir: ${TORCHMLIR_BUILD_DIR}"
