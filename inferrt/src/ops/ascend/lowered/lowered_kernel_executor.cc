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

LoweredKernelExecutor::LoweredKernelExecutor(const std::string &specId)
    : specId_(specId), spec_(nullptr), currentEntry_(nullptr) {
  // Look up kernel spec from registry
  spec_ = KernelRegistry::Instance().Lookup(specId);
  if (spec_ == nullptr) {
    LOG_EXCEPTION << "Kernel spec not found: " << specId;
  }

  LOG_OUT << "LoweredKernelExecutor created for spec: " << specId;
}

LoweredKernelExecutor::~LoweredKernelExecutor() {
  // Cache entries will be cleaned up automatically via unique_ptr
}

std::string LoweredKernelExecutor::GenerateCacheKey(const std::vector<const ir::Value *> &inputs,
                                                    const ir::Value *output) const {
  std::ostringstream oss;
  oss << specId_ << "|inp:" << inputs.size() << "|";

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

  // Lock to prevent race condition during compilation
  {
    std::lock_guard<std::mutex> lock(mutableSpec->compilationMutex_);

    // Check if we need to compile MLIR first
    if (mutableSpec->NeedsCompilation()) {
      LOG_OUT << "Kernel needs compilation from MLIR";

      if (mutableSpec->compiler == nullptr) {
        LOG_ERROR << "No compiler callback provided for MLIR-based kernel";
        return nullptr;
      }

      // Call compiler callback
      std::string outputSoPath, entryName, tilingPrefix;
      bool compileSuccess = mutableSpec->compiler(mutableSpec->mlirText, outputSoPath, entryName, tilingPrefix);

      if (!compileSuccess) {
        LOG_ERROR << "MLIR compilation failed for spec: " << mutableSpec->id;
        return nullptr;
      }

      // Update spec with compiled results
      mutableSpec->kernelLibPath = outputSoPath;
      mutableSpec->entry = entryName;
      mutableSpec->tilingPrefix = tilingPrefix;

      LOG_OUT << "MLIR compilation successful:";
      LOG_OUT << "  - .so path: " << outputSoPath;
      LOG_OUT << "  - entry: " << entryName;
      LOG_OUT << "  - tiling prefix: " << tilingPrefix;
    }
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
    std::string tilingSizeFuncName = mutableSpec->tilingPrefix + "_get_tiling_struct_size_function";
    using GetTilingSizeRawFunc = int64_t (*)();
    auto *getTilingSizeRaw =
      reinterpret_cast<GetTilingSizeRawFunc>(dlsym(entry->dlHandle, tilingSizeFuncName.c_str()));
    if (getTilingSizeRaw == nullptr) {
      LOG_OUT << "Tiling size function not found: " << tilingSizeFuncName << ", assuming no tiling needed";
    } else {
      entry->getTilingSize = getTilingSizeRaw;
      LOG_OUT << "Loaded tiling size function: " << tilingSizeFuncName;
    }

    // Load tiling function
    std::string tilingFuncName = mutableSpec->tilingPrefix + "_tiling_function";
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

int LoweredKernelExecutor::ComputeTiling(LoweredKernelCacheEntry *entry, const std::vector<const ir::Value *> &inputs,
                                         const ir::Value *output) {
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
      AddMemrefArgs(tilingArgs, devicePtr, shape);
    }
  }

  // Add output memref
  const auto &outputTensor = output->ToTensor();
  if (outputTensor != nullptr) {
    void *devicePtr = outputTensor->DataPtr();
    const auto &shape = outputTensor->Shape();
    AddMemrefArgs(tilingArgs, devicePtr, shape);
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

void LoweredKernelExecutor::AddMemrefArgs(std::vector<void *> &args, void *ptr, const std::vector<int64_t> &shape) {
  // MLIR memref lowering to LLVM IR for N-D tensor:
  // struct { T* allocated_ptr, T* aligned_ptr, int64_t offset,
  //          int64_t sizes[N], int64_t strides[N] }

  int64_t offset = 0;
  size_t rank = shape.size();

  // Add allocated_ptr and aligned_ptr
  args.push_back(ptr);
  args.push_back(ptr);
  args.push_back(reinterpret_cast<void *>(offset));

  // Add sizes (dimensions)
  for (size_t i = 0; i < rank; ++i) {
    args.push_back(reinterpret_cast<void *>(shape[i]));
  }

  // Calculate and add strides (row-major layout)
  // stride[i] = product(shape[i+1:])
  for (size_t i = 0; i < rank; ++i) {
    int64_t stride = 1;
    for (size_t j = i + 1; j < rank; ++j) {
      stride *= shape[j];
    }
    args.push_back(reinterpret_cast<void *>(stride));
  }
}

void LoweredKernelExecutor::AddTilingArgs(std::vector<void *> &args, const LoweredKernelCacheEntry *entry) {
  if (entry == nullptr || entry->tilingStructSize <= 0) {
    return;
  }

  int64_t offset = 0;
  args.push_back(const_cast<int64_t *>(&entry->tilingKey));
  args.push_back(entry->dTilingData);
  args.push_back(entry->dTilingData);
  args.push_back(reinterpret_cast<void *>(offset));
  args.push_back(reinterpret_cast<void *>(entry->tilingStructSize));
  args.push_back(reinterpret_cast<void *>(1));
}

void LoweredKernelExecutor::BuildKernelArgs(std::vector<void *> &args, const std::vector<const ir::Value *> &inputs,
                                            const ir::Value *output, const LoweredKernelCacheEntry *entry) {
  args.clear();

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
        AddMemrefArgs(args, devicePtr, shape);
      } else {
        args.push_back(devicePtr);
      }
    }
  }

  // Add output argument
  const auto &outputTensor = output->ToTensor();
  if (outputTensor != nullptr) {
    void *devicePtr = outputTensor->DataPtr();

    if (useMemref) {
      const auto &shape = outputTensor->Shape();
      AddMemrefArgs(args, devicePtr, shape);
    } else {
      args.push_back(devicePtr);
    }
  }

  // Add tiling arguments for dynamic shape
  if (useMemref && entry != nullptr) {
    AddTilingArgs(args, entry);
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
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    auto it = cache_.find(cacheKey);
    if (it != cache_.end()) {
      LOG_OUT << "Cache hit for key: " << cacheKey;
      currentEntry_ = it->second.get();
      *workspaceSize = currentEntry_->workspaceSize;
      return 0;
    }
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
    int ret = ComputeTiling(entry.get(), inputs, output);
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
  {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    cache_[cacheKey] = std::move(entry);
    currentEntry_ = cache_[cacheKey].get();
  }

  *workspaceSize = currentEntry_->workspaceSize;
  LOG_OUT << "Workspace size: " << *workspaceSize;

  return 0;
}

int LoweredKernelExecutor::Launch(void *workspace, size_t workspaceSize, void *stream,
                                  const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  // Use currentEntry_ from GetWorkspaceSize, or look up cache
  LoweredKernelCacheEntry* entry = currentEntry_;

  if (entry == nullptr) {
    // Try to find in cache
    std::string cacheKey = GenerateCacheKey(inputs, output);
    std::lock_guard<std::mutex> lock(cacheMutex_);
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
  BuildKernelArgs(kernelArgs, inputs, output, entry);

  LOG_OUT << "Launching kernel with " << kernelArgs.size() << " args, blockDim=" << entry->blockDim;

  // Call Host API function
  // Signature: void host_api(uint32_t blockDim, void* l2ctrl, void* stream, void** args)
  entry->hostApiFunc(entry->blockDim, nullptr, stream, kernelArgs.data());

  LOG_OUT << "Kernel launched successfully";

  return 0;
}

}  // namespace mrt::ops
