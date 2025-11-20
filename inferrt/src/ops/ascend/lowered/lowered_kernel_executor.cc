/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ops/ascend/lowered/lowered_kernel_executor.h"

#include <dlfcn.h>
#include <sstream>
#include <algorithm>
#include <functional>

#include "acl/acl.h"
#include "common/logger.h"
#include "ir/tensor/tensor.h"
#include "ops/ascend/lowered/mlir_compiler.h"

namespace mrt::ops {

// ============================================================
// LoweredKernelCacheEntry Implementation
// ============================================================

LoweredKernelCacheEntry::~LoweredKernelCacheEntry() {
  // Free device tiling data
  if (dTilingData != nullptr) {
    aclrtFree(dTilingData);
    dTilingData = nullptr;
  }

  // Close dlopen handle
  if (dlHandle != nullptr) {
    dlclose(dlHandle);
    dlHandle = nullptr;
  }
}

// ============================================================
// LoweredKernelExecutor Implementation
// ============================================================

LoweredKernelExecutor::LoweredKernelExecutor(const KernelSpec *spec)
    : spec_(spec), cacheDir_(""), keepIntermediateFiles_(false), currentEntry_(nullptr) {
  if (spec_ == nullptr) {
    LOG_EXCEPTION << "KernelSpec pointer is null";
  }

  // Read keepIntermediateFiles from environment variable
  const char *keepFilesEnv = std::getenv("MRT_LOWERED_MLIR_KEEP_FILES");
  if (keepFilesEnv != nullptr && (std::string(keepFilesEnv) == "1" || std::string(keepFilesEnv) == "true")) {
    keepIntermediateFiles_ = true;
  }
}

LoweredKernelExecutor::~LoweredKernelExecutor() {
  if (!cacheDir_.empty()) {
    if (keepIntermediateFiles_) {
      LOG_OUT << "Keeping compilation cache directory for debugging: " << cacheDir_;
    } else {
      std::string rmCmd = "rm -rf " + cacheDir_;
      int ret = system(rmCmd.c_str());
      (void)ret;
      LOG_OUT << "Cleaned up cache directory: " << cacheDir_;
    }
  }
}

std::string LoweredKernelExecutor::GenerateCacheKey(const std::vector<const ir::Value *> &inputs,
                                                    const ir::Value *output) const {
  std::ostringstream oss;
  oss << spec_->id << "|inp:" << inputs.size() << "|";

  // Add input shapes with explicit null markers
  for (size_t i = 0; i < inputs.size(); ++i) {
    oss << "i" << i << ":";
    const auto &tensor = inputs[i]->ToTensor();
    if (tensor != nullptr) {
      const auto &shape = tensor->Shape();
      // Include dtype to prevent same-shape-different-dtype collisions
      oss << static_cast<int>(tensor->Dtype()) << "#";
      oss << shape.size() << "[";
      for (size_t j = 0; j < shape.size(); ++j) {
        if (j > 0) oss << ",";
        oss << shape[j];
      }
      oss << "]";
    } else {
      oss << "null";
    }
    oss << "|";
  }

  // Add output shape
  oss << "out:";
  const auto &outputTensor = output->ToTensor();
  if (outputTensor != nullptr) {
    const auto &shape = outputTensor->Shape();
    oss << static_cast<int>(outputTensor->Dtype()) << "#";
    oss << shape.size() << "[";
    for (size_t j = 0; j < shape.size(); ++j) {
      if (j > 0) oss << ",";
      oss << shape[j];
    }
    oss << "]";
  } else {
    oss << "null";
  }

  return oss.str();
}

LoweredCacheEntryPtr LoweredKernelExecutor::LoadKernel() {
  if (spec_ == nullptr) {
    LOG_ERROR << "Kernel spec is null";
    return nullptr;
  }

  auto entry = std::make_unique<LoweredKernelCacheEntry>();

  // Get mutable spec pointer for potential compilation
  KernelSpec *mutableSpec = const_cast<KernelSpec *>(spec_);

  // Check if we need to compile MLIR first
  if (mutableSpec->NeedsCompilation()) {
    LOG_OUT << "Kernel needs compilation from MLIR";

    // Prepare compilation request
    MlirCompiler::CompileRequest request = MlirCompiler::internal::InitializeDefaultRequest();
    request.mlirText = mutableSpec->mlirText;
    request.keepIntermediateFiles = keepIntermediateFiles_;

    // Compile MLIR to .so
    MlirCompiler::CompileResult result = MlirCompiler::CompileFromText(request);
    if (!result.success) {
      LOG_ERROR << "MLIR compilation failed: " << result.errorMessage;
      return nullptr;
    }

    // Store cache directory in executor for cleanup (only once)
    cacheDir_ = result.cacheDir;

    // Update spec with compiled results (including id from entryName)
    mutableSpec->kernelLibPath = result.soPath;
    mutableSpec->entry = result.entryName;
    mutableSpec->id = result.entryName;  // Set operator name from compiled entry

    LOG_OUT << "MLIR compilation successful:";
    LOG_OUT << "  - operator: " << mutableSpec->id;
    LOG_OUT << "  - cache dir: " << result.cacheDir;
    LOG_OUT << "  - .so path: " << result.soPath;
    LOG_OUT << "  - entry: " << result.entryName;
  } else {
    LOG_OUT << "Kernel already compiled, using cached binary: " << mutableSpec->kernelLibPath;
  }

  // Check if spec is ready to load
  if (!mutableSpec->IsReadyToLoad()) {
    LOG_ERROR << "Kernel spec is not ready to load (missing .so path)";
    return nullptr;
  }

  // Load dynamic library
  entry->dlHandle = dlopen(mutableSpec->kernelLibPath.c_str(), RTLD_LAZY);
  if (entry->dlHandle == nullptr) {
    LOG_ERROR << "dlopen failed for " << mutableSpec->kernelLibPath << ": " << dlerror();
    return nullptr;
  }

  LOG_OUT << "Loaded kernel library: " << mutableSpec->kernelLibPath;

  // Load Host API function
  using HostApiRawFunc = void (*)(uint32_t, void *, void *, void **);
  auto *hostApiRaw = reinterpret_cast<HostApiRawFunc>(dlsym(entry->dlHandle, mutableSpec->entry.c_str()));
  if (hostApiRaw == nullptr) {
    LOG_ERROR << "dlsym failed for Host API function " << mutableSpec->entry << ": " << dlerror();
    return nullptr;
  }
  entry->hostApiFunc = hostApiRaw;

  LOG_OUT << "Loaded Host API function: " << mutableSpec->entry;

  // For dynamic shape kernels, load tiling functions
  if (mutableSpec->IsDynamicShape()) {
    // Load tiling size function
    std::string tilingSizeFuncName = mutableSpec->entry + "_get_tiling_struct_size_function";
    using GetTilingSizeRawFunc = int64_t (*)();
    auto *getTilingSizeRaw = reinterpret_cast<GetTilingSizeRawFunc>(dlsym(entry->dlHandle, tilingSizeFuncName.c_str()));
    if (getTilingSizeRaw == nullptr) {
      LOG_OUT << "Tiling size function not found: " << tilingSizeFuncName << ", assuming no tiling needed";
    } else {
      entry->getTilingSize = getTilingSizeRaw;
      LOG_OUT << "Loaded tiling size function: " << tilingSizeFuncName;
    }

    // Load tiling function
    std::string tilingFuncName = mutableSpec->entry + "_tiling_function";
    using TilingRawFunc = void (*)(void **);
    auto *tilingRaw = reinterpret_cast<TilingRawFunc>(dlsym(entry->dlHandle, tilingFuncName.c_str()));
    if (tilingRaw == nullptr) {
      LOG_OUT << "Tiling function not found: " << tilingFuncName;
    } else {
      entry->tilingFunc = tilingRaw;
      LOG_OUT << "Loaded tiling function: " << tilingFuncName;
    }
  }

  return entry;
}

int LoweredKernelExecutor::ComputeTiling(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                                         LoweredKernelCacheEntry *entry) {
  if (entry == nullptr) {
    LOG_ERROR << "Cache entry is null";
    return -1;
  }

  // Check if tiling is needed
  if (!entry->getTilingSize || !entry->tilingFunc) {
    LOG_OUT << "No tiling computation needed (functions not available)";
    // TODO(dev): Use device upper limit as default value
    entry->blockDim = 40;  // Default blockDim
    return 0;
  }

  // Get tiling struct size
  entry->tilingStructSize = entry->getTilingSize();
  LOG_OUT << "Tiling struct size: " << entry->tilingStructSize;

  if (entry->tilingStructSize <= 0) {
    LOG_ERROR << "Invalid tiling struct size: " << entry->tilingStructSize
              << " (expected positive value when tiling functions exist)";
    return -1;
  }

  // Allocate host-side tiling data buffer
  entry->tilingData.resize(entry->tilingStructSize, 0);

  // Build tiling function arguments
  // Format: [input memrefs...] [output memrefs...] [tilingKey, tilingBuffer, ...]
  std::vector<void *> tilingArgs;

  // Add input memrefs
  for (const auto *input : inputs) {
    const auto &tensor = input->ToTensor();
    if (tensor != nullptr) {
      void *devicePtr = tensor->DataPtr();
      const auto &shape = tensor->Shape();
      AddMemrefArgs(devicePtr, shape, &tilingArgs);
    }
  }

  // Add output memref
  const auto &outputTensor = output->ToTensor();
  if (outputTensor != nullptr) {
    void *devicePtr = outputTensor->DataPtr();
    const auto &shape = outputTensor->Shape();
    AddMemrefArgs(devicePtr, shape, &tilingArgs);
  }

  // Add tiling arguments
  int64_t offset = 0;
  tilingArgs.push_back(&entry->tilingKey);
  tilingArgs.push_back(entry->tilingData.data());
  tilingArgs.push_back(entry->tilingData.data());
  tilingArgs.push_back(reinterpret_cast<void *>(offset));
  tilingArgs.push_back(reinterpret_cast<void *>(entry->tilingStructSize));
  tilingArgs.push_back(reinterpret_cast<void *>(1));

  // Call tiling function
  LOG_OUT << "Calling tiling function with " << tilingArgs.size() << " args";
  entry->tilingFunc(tilingArgs.data());
  LOG_OUT << "Tiling computation completed, tilingKey: " << entry->tilingKey;

  // Extract blockDim from tiling data
  // Assumes layout: [tile_m, tile_n, blockDim, ...]
  if (entry->tilingStructSize >= 3) {
    entry->blockDim = static_cast<uint32_t>(entry->tilingData[2]);
    LOG_OUT << "Extracted blockDim from tiling data: " << entry->blockDim;
  } else {
    entry->blockDim = 8;
    LOG_OUT << "Using default blockDim: " << entry->blockDim;
  }

  // Validate blockDim range [1, 40]
  if (entry->blockDim < 1 || entry->blockDim > 40) {
    LOG_OUT << "blockDim " << entry->blockDim << " out of range [1, 40], clamping";
    entry->blockDim = std::min(std::max(entry->blockDim, 1u), 40u);
  }

  // Allocate device memory for tiling data
  size_t tilingDataSize = entry->tilingStructSize * sizeof(int64_t);
  aclError error = aclrtMalloc(&entry->dTilingData, tilingDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
  if (error != ACL_RT_SUCCESS) {
    LOG_ERROR << "Failed to allocate device memory for tiling data, error=" << error;
    return -1;
  }

  // Copy tiling data to device
  error = aclrtMemcpy(entry->dTilingData, tilingDataSize, entry->tilingData.data(), tilingDataSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
  if (error != ACL_RT_SUCCESS) {
    LOG_ERROR << "Failed to copy tiling data to device, error=" << error;
    return -1;
  }

  LOG_OUT << "Tiling data copied to device";

  // Set workspace size (tiling data size)
  entry->workspaceSize = tilingDataSize;

  return 0;
}

void LoweredKernelExecutor::AddMemrefArgs(void *ptr, const std::vector<int64_t> &shape, std::vector<void *> *args) {
  // MLIR memref lowering to LLVM IR for N-D tensor:
  // struct { T* allocated_ptr, T* aligned_ptr, int64_t offset,
  //          int64_t sizes[N], int64_t strides[N] }

  int64_t offset = 0;
  size_t rank = shape.size();

  // Add allocated_ptr and aligned_ptr
  args->push_back(ptr);
  args->push_back(ptr);
  args->push_back(reinterpret_cast<void *>(offset));

  // Add sizes (dimensions)
  for (size_t i = 0; i < rank; ++i) {
    args->push_back(reinterpret_cast<void *>(shape[i]));
  }

  // Calculate and add strides (row-major layout)
  // stride[i] = product(shape[i+1:])
  for (size_t i = 0; i < rank; ++i) {
    int64_t stride = 1;
    for (size_t j = i + 1; j < rank; ++j) {
      stride *= shape[j];
    }
    args->push_back(reinterpret_cast<void *>(stride));
  }
}

void LoweredKernelExecutor::AddTilingArgs(const LoweredKernelCacheEntry *entry, std::vector<void *> *args) {
  if (entry == nullptr || entry->tilingStructSize <= 0) {
    return;
  }

  int64_t offset = 0;
  args->push_back(const_cast<int64_t *>(&entry->tilingKey));
  args->push_back(entry->dTilingData);
  args->push_back(entry->dTilingData);
  args->push_back(reinterpret_cast<void *>(offset));
  args->push_back(reinterpret_cast<void *>(entry->tilingStructSize));
  args->push_back(reinterpret_cast<void *>(1));
}

void LoweredKernelExecutor::BuildKernelArgs(const std::vector<const ir::Value *> &inputs, const ir::Value *output,
                                            const LoweredKernelCacheEntry *entry, std::vector<void *> *args) {
  args->clear();

  // For dynamic shape kernels, use memref + tiling args
  // For static shape kernels, just use raw pointers
  bool useMemref = spec_->IsDynamicShape();

  // Add input arguments
  for (const auto *input : inputs) {
    const auto &tensor = input->ToTensor();
    if (tensor != nullptr) {
      void *devicePtr = tensor->DataPtr();

      if (useMemref) {
        const auto &shape = tensor->Shape();
        AddMemrefArgs(devicePtr, shape, args);
      } else {
        args->push_back(devicePtr);
      }
    }
  }

  // Add output argument
  const auto &outputTensor = output->ToTensor();
  if (outputTensor != nullptr) {
    void *devicePtr = outputTensor->DataPtr();

    if (useMemref) {
      const auto &shape = outputTensor->Shape();
      AddMemrefArgs(devicePtr, shape, args);
    } else {
      args->push_back(devicePtr);
    }
  }

  // Add tiling arguments for dynamic shape
  if (useMemref && entry != nullptr) {
    AddTilingArgs(entry, args);
  }
}

int LoweredKernelExecutor::GetWorkspaceSize(size_t *workspaceSize, const std::vector<const ir::Value *> &inputs,
                                            const ir::Value *output) {
  if (workspaceSize == nullptr) {
    LOG_ERROR << "workspaceSize is null";
    return -1;
  }

  // Generate cache key
  std::string cacheKey = GenerateCacheKey(inputs, output);

  // Check cache
  auto it = cache_.find(cacheKey);
  if (it != cache_.end()) {
    LOG_OUT << "Cache hit for key: " << cacheKey;
    currentEntry_ = it->second.get();
    *workspaceSize = currentEntry_->workspaceSize;
    return 0;
  }

  LOG_OUT << "Cache miss for key: " << cacheKey;

  // Load kernel
  auto entry = LoadKernel();
  if (entry == nullptr) {
    LOG_ERROR << "Failed to load kernel";
    return -1;
  }

  // Compute tiling for dynamic shape kernels
  if (spec_->IsDynamicShape()) {
    int ret = ComputeTiling(inputs, output, entry.get());
    if (ret != 0) {
      LOG_ERROR << "Failed to compute tiling";
      return ret;
    }
  } else {
    // Static shape: no workspace needed
    entry->workspaceSize = 0;
    entry->blockDim = 40;  // Use max cores for static shape
  }

  // Cache the entry
  cache_[cacheKey] = std::move(entry);
  currentEntry_ = cache_[cacheKey].get();

  *workspaceSize = currentEntry_->workspaceSize;
  LOG_OUT << "Workspace size: " << *workspaceSize;

  return 0;
}

int LoweredKernelExecutor::Launch(void *workspace, size_t workspaceSize, void *stream,
                                  const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  // Use currentEntry_ from GetWorkspaceSize, or look up cache
  LoweredKernelCacheEntry *entry = currentEntry_;

  if (entry == nullptr) {
    // Try to find in cache
    std::string cacheKey = GenerateCacheKey(inputs, output);
    auto it = cache_.find(cacheKey);
    if (it != cache_.end()) {
      entry = it->second.get();
    } else {
      LOG_ERROR << "Cache entry not found for Launch (GetWorkspaceSize should be called first)";
      return -1;
    }
  }

  if (entry->hostApiFunc == nullptr) {
    LOG_ERROR << "Host API function is null";
    return -1;
  }

  // Build kernel arguments
  std::vector<void *> kernelArgs;
  BuildKernelArgs(inputs, output, entry, &kernelArgs);

  LOG_OUT << "Launching kernel with " << kernelArgs.size() << " args, blockDim=" << entry->blockDim;

  // Call Host API function
  // Signature: void host_api(uint32_t blockDim, void* l2ctrl, void* stream, void** args)
  entry->hostApiFunc(entry->blockDim, nullptr, stream, kernelArgs.data());

  LOG_OUT << "Kernel launched successfully";

  return 0;
}

}  // namespace mrt::ops
