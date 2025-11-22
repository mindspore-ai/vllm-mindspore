#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Find llvm-tblgen executable
LLVM_TBLGEN=""
if [ -f "${LLVM_BUILD_DIR}/bin/llvm-tblgen" ]; then
    LLVM_TBLGEN="${LLVM_BUILD_DIR}/bin/llvm-tblgen"
elif command -v llvm-tblgen >/dev/null 2>&1; then
    LLVM_TBLGEN="llvm-tblgen"
else
    echo "Error: llvm-tblgen not found. Please build LLVM first or install llvm-tblgen."
    exit 1
fi

# Set up include paths for tablegen files
INCLUDE_PATHS=(
    "-I${LLVM_SOURCE_DIR}/mlir/include"
    "-I${PROJECT_DIR}/mopt/include"
)

# Input and output files
# TODO: find .td file in directory
INPUT_FILE="${PROJECT_DIR}/mopt/include/mopt/Dialect/Mrt/Mrt.td"
OUTPUT_FILE="${PROJECT_DIR}/scripts/codegen/MrtDialect.json"

echo "Using llvm-tblgen: ${LLVM_TBLGEN}"
echo "Input file: ${INPUT_FILE}"
echo "Output file: ${OUTPUT_FILE}"

# Run llvm-tblgen to generate JSON
echo "Generating JSON from tablegen file..."
"${LLVM_TBLGEN}" \
    "${INCLUDE_PATHS[@]}" \
    --dump-json \
    -o "${OUTPUT_FILE}" \
    "${INPUT_FILE}"

if [ $? -eq 0 ]; then
    echo "Generate JSON file: ${OUTPUT_FILE} success"
    echo "File size: $(wc -c < "${OUTPUT_FILE}") bytes"
else
    echo "Error: Failed to generate JSON file"
    exit 1
fi

# Run codegen to generate ops code
python "${PROJECT_DIR}/scripts/codegen"
