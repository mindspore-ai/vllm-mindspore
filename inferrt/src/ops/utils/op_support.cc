/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "ops/utils/op_support.h"

#include <algorithm>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "common/logger.h"
#include "hardware/device.h"
#include "nlohmann/json.hpp"
#include "ops/op_register.h"

#if defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#endif

namespace mrt {
namespace runtime {

using json = nlohmann::json;

namespace {

bool IsOpSupportWhitelisted(const std::string &opName) {
  static const std::unordered_set<std::string> whitelist = {
    "make_tuple", "tuple_getitem", "depend", "return", "update_state",
  };
  return whitelist.find(opName) != whitelist.end();
}

// Dialect op info: for all-fixed-tensor ops only; we validate exact input count.
struct DialectOpInfo {
  size_t tensorInputCount{0};
};

std::unordered_map<std::string, DialectOpInfo> &DialectOpInfoMap() {
  static std::unordered_map<std::string, DialectOpInfo> m;
  return m;
}

// Fixed required tensor only (excludes Variadic and Optional).
bool IsMrtFixedTensorType(const std::string &defStr) {
  if (defStr.empty()) {
    return false;
  }
  if (defStr.find("Variadic") != std::string::npos || defStr.find("Optional") != std::string::npos ||
      defStr.find("MrtOptional") != std::string::npos) {
    return false;
  }
  return defStr.find("AnyTensor") != std::string::npos || defStr.find("TensorList") != std::string::npos;
}

// Extract "def" from type spec; may be nested in tablegen output.
std::string GetDefString(const json &typeSpec) {
  if (typeSpec.is_string()) {
    return typeSpec.get<std::string>();
  }
  if (typeSpec.is_object() && typeSpec.contains("def")) {
    const auto &defVal = typeSpec["def"];
    if (defVal.is_string()) {
      return defVal.get<std::string>();
    }
  }
  if (typeSpec.is_object() && typeSpec.contains("printable")) {
    const auto &p = typeSpec["printable"];
    if (p.is_string()) {
      return p.get<std::string>();
    }
  }
  return "";
}

// Find MrtDialect.json relative to lib (share/mrt/MrtDialect.json).
std::string FindMrtDialectJsonPath() {
#if defined(__linux__) || defined(__APPLE__)
  Dl_info dlInfo;
  if (dladdr(reinterpret_cast<void *>(&FindMrtDialectJsonPath), &dlInfo) != 0 && dlInfo.dli_fname != nullptr) {
    std::string libPath(dlInfo.dli_fname);
    size_t slash = libPath.find_last_of("/\\");
    if (slash != std::string::npos) {
      std::string libDir = libPath.substr(0, slash);
      std::string candidate = libDir + "/share/mrt/MrtDialect.json";
      std::ifstream f(candidate);
      if (f.good()) {
        return candidate;
      }
      LOG_ERROR << "MrtDialect.json not found or unreadable at: " << candidate;
    } else {
      LOG_ERROR << "Cannot get lib directory from path (no directory separator): " << libPath;
    }
  } else {
    LOG_ERROR << "dladdr failed or dli_fname is null, cannot resolve MrtDialect.json path";
  }
#else
  LOG_ERROR << "findMrtDialectJsonPath is only supported on Linux and macOS";
#endif
  return "";
}

void LoadMrtDialectJson(const std::string &path) {
  std::ifstream f(path);
  if (!f.is_open()) {
    LOG_ERROR << "Cannot open MrtDialect.json: " << path;
    return;
  }
  json j;
  try {
    f >> j;
  } catch (const json::exception &e) {
    LOG_ERROR << "Failed to parse MrtDialect.json: " << e.what();
    return;
  }
  if (!j.is_object()) {
    return;
  }

  auto &dialectMap = DialectOpInfoMap();
  for (auto it = j.begin(); it != j.end(); ++it) {
    const json &opDef = it.value();
    if (!opDef.is_object()) {
      continue;
    }
    std::string opName;
    if (opDef.contains("opName") && opDef["opName"].is_string()) {
      opName = opDef["opName"].get<std::string>();
    } else {
      continue;
    }
    if (!opDef.contains("arguments") || !opDef["arguments"].is_object()) {
      continue;
    }
    const auto &args = opDef["arguments"];
    if (!args.contains("args") || !args["args"].is_array()) {
      continue;
    }
    const auto &argsArr = args["args"];

    size_t tensorCount = 0;
    bool skipOp = false;

    for (const auto &argEntry : argsArr) {
      if (!argEntry.is_array() || argEntry.empty()) {
        continue;
      }
      const json &typeSpec = argEntry[0];
      std::string defStr = GetDefString(typeSpec);

      if (IsMrtFixedTensorType(defStr)) {
        tensorCount += 1;
      } else {
        skipOp = true;
        break;
      }
    }

    if (skipOp || tensorCount == 0) {
      continue;
    }
    DialectOpInfo info;
    info.tensorInputCount = tensorCount;
    dialectMap[opName] = info;
  }
}

const std::unordered_map<std::string, DialectOpInfo> &GetDialectOpInfoMap() {
  static bool loaded = false;
  if (!loaded) {
    loaded = true;
    std::string path = FindMrtDialectJsonPath();
    if (!path.empty()) {
      LoadMrtDialectJson(path);
    }
  }
  return DialectOpInfoMap();
}

bool IsOpRegisteredOnDevice(const std::string &opName, const hardware::DeviceType deviceType) {
  if (deviceType == hardware::DeviceType::NPU) {
    return mrt::ops::OpFactory<mrt::ops::Operator>::GetInstance().IsRegistered(opName);
  }
  if (deviceType == hardware::DeviceType::CPU) {
    return mrt::ops::OpFactory<mrt::ops::Operator, mrt::ops::CPUOpFactory>::GetInstance().IsRegistered(opName);
  }
  return false;
}

// For all-tensor ops: verify all actual inputs are tensors and count matches prototype.
OpSupportResult CheckInputTypesSupported(const std::string &opName, const std::vector<ir::ValuePtr> &inputValues) {
  const auto &dialectMap = GetDialectOpInfoMap();
  auto dialIt = dialectMap.find(opName);
  if (dialIt == dialectMap.end()) {
    OpSupportResult r;
    r.status = OpSupportStatus::kOk;
    r.message.clear();
    return r;
  }

  const DialectOpInfo &info = dialIt->second;

  // Some graphs carry control-edge dependencies as `None` values (e.g. output of `End`).
  // For all-fixed-tensor ops, these `None` inputs are not real data inputs and should be ignored.
  std::vector<ir::ValuePtr> filteredInputs;
  filteredInputs.reserve(inputValues.size());
  for (const auto &v : inputValues) {
    if (v != nullptr && v->IsNone()) {
      continue;
    }
    filteredInputs.push_back(v);
  }

  // 1. Verify all actual inputs are tensors (no scalar, tuple, etc.)
  for (size_t i = 0; i < filteredInputs.size(); ++i) {
    const auto &v = filteredInputs[i];
    if (v == nullptr || !v->IsTensor()) {
      OpSupportResult r;
      r.status = OpSupportStatus::kUnsupportedInputType;
      r.message =
        "operator '" + opName + "' expects all tensor inputs, but input[" + std::to_string(i) + "] is not a tensor";
      return r;
    }
  }

  // 2. Verify input count matches prototype (exact match for all-fixed-tensor ops)
  size_t actualCount = filteredInputs.size();
  if (actualCount != info.tensorInputCount) {
    OpSupportResult r;
    r.status = OpSupportStatus::kUnsupportedInputType;
    r.message = "operator '" + opName + "' expects " + std::to_string(info.tensorInputCount) +
                " tensor input(s), got " + std::to_string(actualCount);
    return r;
  }

  OpSupportResult r;
  r.status = OpSupportStatus::kOk;
  r.message.clear();
  return r;
}

}  // namespace

hardware::Device GetDeviceFromOutputAndInputs(const ir::ValuePtr &output, const std::vector<ir::ValuePtr> &inputs) {
  CHECK_IF_NULL(output);

  if (output->IsTensor()) {
    auto &tensor = output->ToTensor();
    CHECK_IF_NULL(tensor);
    return tensor->GetDevice();
  }

  if (output->IsNone()) {
    auto it =
      std::find_if(inputs.begin(), inputs.end(), [](const ir::ValuePtr &v) { return v != nullptr && v->IsTensor(); });
    if (it != inputs.end()) {
      return (*it)->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  if (output->IsTuple()) {
    auto &tuple = output->ToTuple();
    CHECK_IF_NULL(tuple);

    if (tuple->Size() == 0) {
      return {hardware::DeviceType::CPU, 0};
    }

    bool allTensor = std::all_of(tuple->begin(), tuple->end(),
                                 [](const ir::ValuePtr &elem) { return elem != nullptr && elem->IsTensor(); });

    if (allTensor) {
      return (*tuple->begin())->ToTensor()->GetDevice();
    }
    return {hardware::DeviceType::CPU, 0};
  }

  return {hardware::DeviceType::CPU, 0};
}

OpSupportResult CheckOpSupport(const std::string &opName, const ir::ValuePtr &outputValue,
                               const std::vector<ir::ValuePtr> &inputValues) {
  if (IsOpSupportWhitelisted(opName)) {
    OpSupportResult r;
    r.status = OpSupportStatus::kOk;
    r.message.clear();
    return r;
  }

  const hardware::Device device = GetDeviceFromOutputAndInputs(outputValue, inputValues);
  if (!IsOpRegisteredOnDevice(opName, device.type)) {
    OpSupportResult r;
    r.status = OpSupportStatus::kUnsupportedDevice;
    r.message =
      "operator '" + opName + "' not registered on target device '" + hardware::GetDeviceNameByType(device.type) + "'";
    return r;
  }

  return CheckInputTypesSupported(opName, inputValues);
}

}  // namespace runtime
}  // namespace mrt
