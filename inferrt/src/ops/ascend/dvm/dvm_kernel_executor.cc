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

#include "ops/ascend/dvm/dvm_kernel_executor.h"

#include <cstdlib>
#include <string_view>

#include "common/logger.h"
#include "common/throw.h"
#include "ir/tensor/tensor.h"

namespace mrt::ops {

namespace {
inline char ToLowerAscii(char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

inline bool EqualsIgnoreCase(std::string_view a, std::string_view b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (ToLowerAscii(a[i]) != ToLowerAscii(b[i])) return false;
  }
  return true;
}

bool GetEnvBoolOrDefault(const char *name, bool default_value) {
  const char *val = std::getenv(name);
  if (val == nullptr) return default_value;
  std::string_view s(val);
  if (s.empty()) return default_value;

  if (s == "1" || EqualsIgnoreCase(s, "true") || EqualsIgnoreCase(s, "on") || EqualsIgnoreCase(s, "yes")) {
    return true;
  }
  if (s == "0" || EqualsIgnoreCase(s, "false") || EqualsIgnoreCase(s, "off") || EqualsIgnoreCase(s, "no")) {
    return false;
  }

  LOG_ERROR << "Invalid env '" << name << "'='" << s << "', using default " << default_value;
  return default_value;
}

// DVM library initialization (called once)
bool InitDvmLibrary() {
  static bool initialized = false;
  if (!initialized) {
    // Set DVM configuration - these calls are required before using DVM.
    //
    // TODO(lmy): replace env var based config with MRT runtime config knobs (align with MindSpore MsContext/flags).
    constexpr bool kDefaultDeterministic = false;
    constexpr bool kDefaultOnlineTuning = true;
    const bool enable_deterministic = GetEnvBoolOrDefault("MRT_DVM_DETERMINISTIC", kDefaultDeterministic);
    bool enable_tuning = GetEnvBoolOrDefault("MRT_DVM_ONLINE_TUNING", kDefaultOnlineTuning);
    if (enable_deterministic && enable_tuning) {
      enable_tuning = false;
      LOG_ERROR << "MRT_DVM_DETERMINISTIC is enabled, forcing MRT_DVM_ONLINE_TUNING=false";
    }
    dvm::SetDeterministic(enable_deterministic);
    dvm::SetOnlineTuning(enable_tuning);
    initialized = true;
    LOG_OUT << "DVM library initialized";
  }
  return initialized;
}
}  // namespace

DvmKernelExecutor::DvmKernelExecutor(dvm::KernelType kernelType)
    : kernelType_(kernelType), kernel_(), isCodeGenDone_(false), cachedWorkspaceSize_(0), isKernelBuilt_(false) {
  // Ensure DVM library is initialized before use
  InitDvmLibrary();

  // Initialize kernel with specified type
  kernel_.Reset(kernelType_);
  LOG_OUT << "DvmKernelExecutor created with kernel type: " << static_cast<int>(kernelType_);
}

DvmKernelExecutor::~DvmKernelExecutor() { LOG_OUT << "DvmKernelExecutor destroyed"; }

void DvmKernelExecutor::EnsureShapeRefsInitialized(size_t numInputs) {
  // NOTE: DvmKernelExecutor currently only supports a single output shape ref.
  // TODO(lmy): support multi-output (tuple) by plumbing output count & output ShapeRef pointers through build/launch.
  constexpr size_t kNumOutputs = 1;
  const size_t expected = numInputs + kNumOutputs;  // inputs + output(s)
  if (!shapeRefs_.empty()) {
    // Once the kernel graph is built, DVM may keep pointers to ShapeRef objects.
    // Do NOT reallocate/move ShapeRefs after that point.
    if (shapeRefs_.size() != expected) {
      MRT_THROW("ShapeRef count mismatch. Expected ", expected, ", got ", shapeRefs_.size());
    }
    return;
  }

  shapesStorage_.resize(expected);
  shapeRefs_.resize(expected);
  inputShapeRefPtrs_.resize(numInputs);

  for (size_t i = 0; i < numInputs; ++i) {
    shapeRefs_[i] = shapesStorage_[i];
    inputShapeRefPtrs_[i] = &shapeRefs_[i];
  }
  shapeRefs_[numInputs] = shapesStorage_[numInputs];
  outputShapeRefPtr_ = &shapeRefs_[numInputs];
}

void DvmKernelExecutor::UpdateShapeRefs(const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  EnsureShapeRefsInitialized(inputs.size());

  if (shapeRefs_.size() != inputs.size() + 1) {
    // Defensive: avoid writing out of bounds.
    LOG_EXCEPTION << "UpdateShapeRefs skipped due to ShapeRef size mismatch";
    return;
  }

  LOG_OUT << "UpdateShapeRefs: processing " << inputs.size() << " inputs";

  // Update input shapes in-place.
  //
  // IMPORTANT for dyn-shape:
  // - DVM may cache/copy ShapeRef (or even shape data pointer) during graph build.
  // - If we reallocate the backing storage of the shape vector after BuildKernel,
  //   those cached pointers can become dangling and crash during CodeGen/Launch.
  // Therefore, we must keep the *address* of each shape buffer stable after the
  // first initialization: only mutate values in-place, do not assign/resize.
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &tensor = inputs[i]->ToTensor();
    if (tensor == nullptr) {
      LOG_ERROR << "Input value is not a tensor";
      shapesStorage_[i].clear();
      shapeRefs_[i] = shapesStorage_[i];  // refresh pointers/size (data may be null)
    } else {
      const auto &shape = tensor->Shape();
      // Log input shape for debugging
      std::string shapeStr = "[";
      for (size_t d = 0; d < shape.size(); ++d) {
        shapeStr += std::to_string(shape[d]);
        if (d < shape.size() - 1) shapeStr += ", ";
      }
      shapeStr += "]";
      LOG_OUT << "Input[" << i << "] shape: " << shapeStr;

      if (shapesStorage_[i].empty()) {
        // First initialization (allowed to allocate).
        shapesStorage_[i].assign(shape.begin(), shape.end());
      } else if (shapesStorage_[i].size() != shape.size()) {
        // Rank change would require resizing/reallocation which can invalidate
        // cached pointers inside DVM. Fail fast with a readable error.
        LOG_EXCEPTION << "DVM dyn-shape rank change is not supported. " << "Input[" << i << "] rank changed from "
                      << shapesStorage_[i].size() << " to " << shape.size() << ". " << "Old shape=" << shapesStorage_[i]
                      << ", new shape=" << shape;
      } else {
        // In-place update (no reallocation).
        for (size_t d = 0; d < shape.size(); ++d) {
          shapesStorage_[i][d] = shape[d];
        }
      }
      // Refresh ShapeRef to current (stable) buffer.
      shapeRefs_[i] = shapesStorage_[i];
    }
  }

  // Update output shape
  const auto &outputTensor = output->ToTensor();
  if (outputTensor == nullptr) {
    LOG_ERROR << "Output value is not a tensor";
    shapesStorage_.back().clear();
    shapeRefs_.back() = shapesStorage_.back();
  } else {
    const auto &outShape = outputTensor->Shape();
    // Log output shape for debugging
    std::string outShapeStr = "[";
    for (size_t d = 0; d < outShape.size(); ++d) {
      outShapeStr += std::to_string(outShape[d]);
      if (d < outShape.size() - 1) outShapeStr += ", ";
    }
    outShapeStr += "]";
    LOG_OUT << "Output shape: " << outShapeStr;
    if (shapesStorage_.back().empty()) {
      // First initialization (allowed to allocate).
      shapesStorage_.back().assign(outShape.begin(), outShape.end());
    } else if (shapesStorage_.back().size() != outShape.size()) {
      LOG_EXCEPTION << "DVM dyn-shape rank change is not supported. " << "Output rank changed from "
                    << shapesStorage_.back().size() << " to " << outShape.size() << ". "
                    << "Old shape=" << shapesStorage_.back() << ", new shape=" << outShape;
    } else {
      for (size_t d = 0; d < outShape.size(); ++d) {
        shapesStorage_.back()[d] = outShape[d];
      }
    }
    shapeRefs_.back() = shapesStorage_.back();
  }

  LOG_OUT << "Updated shape refs: " << shapeRefs_.size() << " shapes";
}

int DvmKernelExecutor::BuildKernel(
  std::function<void(dvm::Kernel &, const std::vector<const ir::Value *> &, const ir::Value *,
                     const std::vector<dvm::ShapeRef *> &, const dvm::ShapeRef *, std::vector<dvm::NDObject *> *,
                     std::vector<dvm::NDObject *> *)>
    buildFunc,
  const std::vector<const ir::Value *> &inputs, const ir::Value *output) {
  if (isKernelBuilt_) {
    LOG_OUT << "Kernel already built, skipping rebuild";
    return 0;
  }

  // Update shape references before building
  UpdateShapeRefs(inputs, output);

  // Clear previous NDObject records
  inputNDObjects_.clear();
  outputNDObjects_.clear();

  // Call user-provided build function with NDObject* output parameters
  buildFunc(kernel_, inputs, output, inputShapeRefPtrs_, outputShapeRefPtr_, &inputNDObjects_, &outputNDObjects_);

  // Validate that NDObjects were properly recorded
  if (inputNDObjects_.size() != inputs.size()) {
    LOG_ERROR << "BuildFunc did not record correct number of input NDObjects. " << "Expected " << inputs.size()
              << ", got " << inputNDObjects_.size();
    return -1;
  }

  if (outputNDObjects_.empty()) {
    LOG_ERROR << "BuildFunc did not record any output NDObjects";
    return -1;
  }
  if (outputNDObjects_.size() != 1) {
    // NOTE: current executor only supports a single output address + single output ShapeRef.
    // TODO(lmy): support multi-output (tuple) by extending Launch/RelocTable/address mapping for N outputs.
    MRT_THROW("DVM kernel currently supports single output only, but got ", outputNDObjects_.size(),
              " output NDObjects. Please ensure payload has exactly one output Store or implement multi-output.");
  }

  isKernelBuilt_ = true;
  LOG_OUT << "DVM kernel graph built successfully with " << inputNDObjects_.size() << " inputs, "
          << outputNDObjects_.size() << " outputs";
  return 0;
}

int DvmKernelExecutor::GetWorkspaceSize(size_t *workspaceSize, const std::vector<const ir::Value *> &inputs,
                                        const ir::Value *output) {
  if (workspaceSize == nullptr) {
    LOG_ERROR << "workspaceSize pointer is null";
    return -1;
  }

  if (!isKernelBuilt_) {
    LOG_ERROR << "Kernel not built yet, call BuildKernel first";
    return -1;
  }

  const bool isStaticKernel = (kernelType_ == dvm::kStaticShape || kernelType_ == dvm::kStaticMix ||
                               kernelType_ == dvm::kStaticParallel || kernelType_ == dvm::kStaticStages);
  const bool isDynKernel = (kernelType_ == dvm::kDynShape || kernelType_ == dvm::kDynMix);

  // For static kernels, return cached result if available
  if (isStaticKernel && isCodeGenDone_) {
    *workspaceSize = cachedWorkspaceSize_;
    LOG_OUT << "Returning cached workspace size: " << *workspaceSize;
    return 0;
  }

  // For dynamic-shape kernels, update shapes before CodeGen.
  if (isDynKernel) {
    UpdateShapeRefs(inputs, output);
  }

  // Call DVM CodeGen
  LOG_OUT << "Calling DVM CodeGen...";
  LOG_OUT << "DVM kernel dump before CodeGen:\n" << kernel_.Dump();

  // MindSpore does not rely on Infer() to "fix" static MatMul shapes; correct kernel type (*Mix)
  // is the real fix. We keep Infer() only for dynamic kernels for debugging/shape inference.
  if (isDynKernel) {
    kernel_.Infer();
    LOG_OUT << "DVM kernel dump after Infer:\n" << kernel_.Dump();
  }

  uint64_t ws = kernel_.CodeGen();
  cachedWorkspaceSize_ = ws;
  isCodeGenDone_ = true;
  *workspaceSize = static_cast<size_t>(ws);

  LOG_OUT << "DVM CodeGen completed, workspace size: " << *workspaceSize;
  return 0;
}

void DvmKernelExecutor::BuildRelocTable(const std::vector<const ir::Value *> &inputs, ir::Value *output) {
  // Validate that kernel was built and NDObjects were recorded
  if (inputNDObjects_.size() != inputs.size()) {
    LOG_ERROR << "Mismatch between recorded NDObjects and actual inputs. " << "NDObjects: " << inputNDObjects_.size()
              << ", Inputs: " << inputs.size();
  }

  if (outputNDObjects_.empty()) {
    LOG_ERROR << "No output NDObjects recorded during kernel build";
  }

  // Build RelocTable using NDObject pointers from BuildKernel phase
  // These pointers were returned by kernel_.Load() and kernel_.Store()
  relocTable_.inputs = inputNDObjects_.data();
  relocTable_.inputs_size = inputNDObjects_.size();
  relocTable_.outputs = outputNDObjects_.data();
  relocTable_.outputs_size = outputNDObjects_.size();

  LOG_OUT << "RelocTable built with " << relocTable_.inputs_size << " input NDObjects, " << relocTable_.outputs_size
          << " output NDObjects";
}

int DvmKernelExecutor::Launch(void *workspace, size_t workspaceSize, void *stream,
                              const std::vector<const ir::Value *> &inputs, ir::Value *output) {
  if (!isKernelBuilt_) {
    LOG_ERROR << "Kernel not built yet";
    return -1;
  }

  if (!isCodeGenDone_) {
    LOG_ERROR << "CodeGen not done yet, call GetWorkspaceSize first";
    return -1;
  }

  // Build relocation table with actual device addresses
  BuildRelocTable(inputs, output);

  // Prepare input/output device address arrays
  std::vector<void *> inputAddrs;
  std::vector<void *> outputAddrs;

  inputAddrs.reserve(inputs.size());
  for (const auto *input : inputs) {
    const auto &tensor = input->ToTensor();
    inputAddrs.push_back(tensor ? tensor->DataPtr() : nullptr);
  }

  const auto &outputTensor = output->ToTensor();
  outputAddrs.push_back(outputTensor ? outputTensor->DataPtr() : nullptr);

  // Launch DVM kernel
  int ret = kernel_.Launch(relocTable_, inputAddrs.data(), outputAddrs.data(), workspace, stream);

  if (ret != 0) {
    LOG_ERROR << "DVM kernel launch failed with code: " << ret;
    return ret;
  }

  LOG_OUT << "DVM kernel launched successfully";
  return 0;
}

}  // namespace mrt::ops
