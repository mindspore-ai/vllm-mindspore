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

# Apply patches to torch-mlir
echo "Applying patches to torch-mlir..."
TORCHMLIR_PATCH_DIR="${PROJECT_DIR}/third_party/patch/torch-mlir"
if [ -d "${TORCHMLIR_PATCH_DIR}" ]; then
  # Apply patches in order: [0-9][0-9][0-9]-*.patch
  for patch_file in "${TORCHMLIR_PATCH_DIR}"/[0-9][0-9][0-9]-*.patch; do
    if [ -f "${patch_file}" ]; then
      echo "  Applying patch: $(basename ${patch_file})"
      patch -p1 -d "${TORCHMLIR_DIR}" < "${patch_file}"
    fi
  done
  echo "Patches applied successfully."
else
  echo "  No patches found, skipping."
fi

#------------ build llvm with torch_mlir in-tree
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${PROJECT_DIR}/build/third_party/build/llvm}"

cd "${PROJECT_DIR}"
echo "Configuring LLVM with torch-mlir in-tree..."
cmake -GNinja -B "${LLVM_BUILD_DIR}" \
    ${CCACHE_CMAKE_ARGS} \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython3_FIND_VIRTUALENV=ONLY \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS="torch-mlir" \
    -DLLVM_EXTERNAL_TORCH_MLIR_SOURCE_DIR="${TORCHMLIR_DIR}" \
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DTORCH_MLIR_ENABLE_ONLY_MLIR_PYTHON_BINDINGS=ON \
    third_party/llvm-project/llvm
echo "Configured LLVM with torch-mlir."

echo "Building LLVM and torch-mlir..."
if [ -n "${BUILD_JOBS}" ]; then
  cmake --build "${LLVM_BUILD_DIR}" -j "${BUILD_JOBS}"
else
  cmake --build "${LLVM_BUILD_DIR}"
fi
echo "Built LLVM and torch-mlir: ${LLVM_BUILD_DIR}"

echo ""
echo "=========================================="
echo "LLVM and torch-mlir build completed"
echo "=========================================="
echo "LLVM build directory: ${LLVM_BUILD_DIR}"