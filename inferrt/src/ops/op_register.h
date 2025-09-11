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

#ifndef __OPS_OP_REGISTER_H__
#define __OPS_OP_REGISTER_H__

#include <utility>
#include <string>
#include <string_view>
#include <memory>
#include <functional>
#include <type_traits>
#include <unordered_map>

#include "common/logger.h"
#include "common/common.h"
#include "ops/operator.h"

namespace mrt {
namespace ops {
inline constexpr std::string_view kUnknownOpFactory = "Unknown";
inline constexpr std::string_view kAscendOpFactory = "Ascend";
inline constexpr std::string_view kCPUOpFactory = "CPU";

struct UnknownOpFactory {};
struct AscendOpFactory {};
struct CPUOpFactory {};

template <typename T>
struct OpFactoryTraits;

template <>
struct OpFactoryTraits<UnknownOpFactory> {
  static constexpr std::string_view value = kUnknownOpFactory;
};

template <>
struct OpFactoryTraits<AscendOpFactory> {
  static constexpr std::string_view value = kAscendOpFactory;
};

template <>
struct OpFactoryTraits<CPUOpFactory> {
  static constexpr std::string_view value = kCPUOpFactory;
};

class DA_API OpFactoryBase {
  using OpFactoryMapType = std::unordered_map<std::string_view, std::unique_ptr<OpFactoryBase>>;

 public:
  OpFactoryBase() = default;
  virtual ~OpFactoryBase() = default;

 protected:
  static OpFactoryBase *GetOpFactory(const std::string_view &name);

  static OpFactoryBase *CreateOpFactory(const std::string_view &name, std::unique_ptr<OpFactoryBase> &&factory);

 private:
  static OpFactoryMapType &OpFactoryMap();
};

template <typename T, typename OpFactoryType = AscendOpFactory>
class OpFactory : public OpFactoryBase {
  using CreatorFunc = std::function<std::unique_ptr<T>()>;

 public:
  OpFactory() = default;
  ~OpFactory() = default;
  OpFactory(const OpFactory &) = delete;
  void operator=(const OpFactory &) = delete;

  static OpFactory<T, OpFactoryType> &GetInstance() {
    auto factoryBase = OpFactoryBase::GetOpFactory(OpFactoryTraits<OpFactoryType>::value);
    if (factoryBase == nullptr) {
      factoryBase = OpFactoryBase::CreateOpFactory(OpFactoryTraits<OpFactoryType>::value,
                                                   std::make_unique<OpFactory<T, OpFactoryType>>());
    }
    return *static_cast<OpFactory<T, OpFactoryType> *>(factoryBase);
  }

  void Register(const std::string &opName, CreatorFunc &&creator) {
    if (IsRegistered(opName)) {
      LOG_EXCEPTION << "Repeat register for op " << opName;
    }
    (void)opCreatorsMap_.emplace(opName, std::move(creator));
  }

  void UnRegister(const std::string &opName) {
    auto iter = opCreatorsMap_.find(opName);
    if (iter != opCreatorsMap_.end()) {
      opCreatorsMap_.erase(iter);
    }
  }

  bool IsRegistered(const std::string &opName) const { return opCreatorsMap_.find(opName) != opCreatorsMap_.end(); }

  std::unique_ptr<T> Create(const std::string &opName) const {
    typename std::unordered_map<std::string, CreatorFunc>::const_iterator iter = opCreatorsMap_.find(opName);
    if (iter != opCreatorsMap_.cend()) {
      return (iter->second)();
    }
    LOG_EXCEPTION << "Operator not registered: " << opName;
    return nullptr;
  }

 private:
  std::unordered_map<std::string, CreatorFunc> opCreatorsMap_;
};

template <typename T, typename OpFactoryType = AscendOpFactory>
class OpRegistrar {
 public:
  explicit OpRegistrar(const std::string &opName, std::function<std::unique_ptr<T>()> creator) {
    OpFactory<T, OpFactoryType>::GetInstance().Register(opName, std::move(creator));
  }
  ~OpRegistrar() = default;
};

#define MRT_REG_OP(OP_NAME, OP_CLASS, DEVICE_NAME)                                                                  \
  static_assert(std::is_base_of<ops::Operator, OP_CLASS>::value, #OP_CLASS " must be derived from class Operator"); \
  static const ops::OpRegistrar<ops::Operator, ops::DEVICE_NAME##OpFactory>                                         \
    g_##OP_NAME##_##OP_CLASS##_##DEVICE_NAME##_reg(#OP_NAME, []() { return std::make_unique<OP_CLASS>(); })

#define MRT_REG_OP_WITH_CREATOR(OP_NAME, OP_CLASS, DEVICE_NAME, CREATOR)                                            \
  static_assert(std::is_base_of<ops::Operator, OP_CLASS>::value, #OP_CLASS " must be derived from class Operator"); \
  static const ops::OpRegistrar<ops::Operator, ops::DEVICE_NAME##OpFactory>                                         \
    g_##OP_NAME##_##OP_CLASS##_##DEVICE_NAME##_reg(#OP_NAME, CREATOR)

inline std::unique_ptr<Operator> CreateOperator(const std::string &name) {
  auto op = OpFactory<Operator>::GetInstance().Create(name);
  if (op == nullptr) {
    op = OpFactory<Operator, CPUOpFactory>::GetInstance().Create(name);
  }
  if (op == nullptr) {
    LOG_EXCEPTION << "Failed to create operator [" << name << "], maybe it has not been registered";
  }
  return op;
}

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_OP_REGISTER_H__
