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

#ifndef __OPS_CUSTOM_OP_REGISTER_H__
#define __OPS_CUSTOM_OP_REGISTER_H__

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "ops/operator.h"

namespace mrt {
namespace ops {

/**
 * @brief Universal custom operator registry
 *
 * Provides dynamic registration of custom operators across different device types (CPU, NPU, GPU,
 * etc.) without relying on static configuration files like ops.list. Supports runtime dynamic
 * loading and registration of custom operators.
 */
class DA_API CustomOpRegistry {
 public:
  using CustomOpFactory = std::function<std::unique_ptr<Operator>()>;

  /**
   * @brief Get singleton instance
   */
  static CustomOpRegistry &GetInstance();

  /**
   * @brief Register custom operator
   * @param op_name Operator name
   * @param factory Operator factory function
   * @return Whether registration was successful
   */
  bool RegisterCustomOp(const std::string &op_name, CustomOpFactory &&factory);

  /**
   * @brief Create custom operator instance
   * @param op_name Operator name
   * @return Operator instance, returns nullptr if not found
   */
  std::unique_ptr<Operator> CreateCustomOp(const std::string &op_name);

  /**
   * @brief Check if operator is registered
   * @param op_name Operator name
   * @return Whether it is registered
   */
  bool IsCustomOpRegistered(const std::string &op_name) const;

  /**
   * @brief Get all registered custom operator names
   * @return List of operator names
   */
  std::vector<std::string> GetRegisteredOpNames() const;

  /**
   * @brief Unregister custom operator
   * @param op_name Operator name
   * @return Whether unregistration was successful
   */
  bool UnregisterCustomOp(const std::string &op_name);

 private:
  CustomOpRegistry() = default;
  ~CustomOpRegistry() = default;

  // Disable copy and assignment
  CustomOpRegistry(const CustomOpRegistry &) = delete;
  CustomOpRegistry &operator=(const CustomOpRegistry &) = delete;

  std::unordered_map<std::string, CustomOpFactory> custom_ops_;
};

DA_API std::unique_ptr<Operator> CreateCustomOperator(const std::string &name);

/**
 * @brief Macro for registering custom operators
 *
 * Usage example:
 * REGISTER_CUSTOM_OP(custom_add, CustomAddOp)
 */
#define REGISTER_CUSTOM_OP(OP_NAME, OP_CLASS)                                                                         \
  static bool g_##OP_NAME##_##OP_CLASS##_registered = []() {                                                          \
    return CustomOpRegistry::GetInstance().RegisterCustomOp(#OP_NAME, []() { return std::make_unique<OP_CLASS>(); }); \
  }()

/**
 * @brief Macro for registering custom operators with custom factory
 *
 * Usage example:
 * REGISTER_CUSTOM_OP_WITH_FACTORY(custom_add, CustomAddOp, []() { return
 * std::make_unique<CustomAddOp>(arg1, arg2); })
 */
#define REGISTER_CUSTOM_OP_WITH_FACTORY(OP_NAME, OP_CLASS, FACTORY)             \
  static bool g_##OP_NAME##_##OP_CLASS##_registered = []() {                    \
    return CustomOpRegistry::GetInstance().RegisterCustomOp(#OP_NAME, FACTORY); \
  }()

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_CUSTOM_OP_REGISTER_H__
