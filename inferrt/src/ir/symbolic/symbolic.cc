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

#include <cmath>

#include "common/common.h"
#include "ir/symbolic/symbolic.h"

namespace mrt {
namespace ir {

int64_t SymbolicVar::Evaluate() const {
  if (value_ < 0) {
    LOG_EXCEPTION << "Symbolic variable " << name_ << " has no value.";
  }
  return value_;
}

int64_t SymbolicFloorDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    LOG_EXCEPTION << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(std::floor(static_cast<double>(lhsVal) / static_cast<double>(rhsVal)));
}

int64_t SymbolicCeilDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    LOG_EXCEPTION << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(std::ceil(static_cast<double>(lhsVal) / static_cast<double>(rhsVal)));
}

int64_t SymbolicTrueDiv::Evaluate() const {
  auto lhsVal = lhs_->Evaluate();
  auto rhsVal = rhs_->Evaluate();
  if (rhsVal == 0) {
    LOG_EXCEPTION << "Division by zero in symbolic expression.";
  }
  return static_cast<int64_t>(static_cast<double>(lhsVal) / static_cast<double>(rhsVal));
}

SymbolicExprPtr operator+(SymbolicExprPtr lhs, SymbolicExprPtr rhs) { return MakeIntrusive<SymbolicAdd>(lhs, rhs); }

SymbolicExprPtr operator*(SymbolicExprPtr lhs, SymbolicExprPtr rhs) { return MakeIntrusive<SymbolicMul>(lhs, rhs); }

SymbolicExprPtr operator/(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicTrueDiv>(lhs, rhs);
}

SymbolicExprPtr FloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicFloorDiv>(lhs, rhs);
}

SymbolicExprPtr CeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicCeilDiv>(lhs, rhs);
}

}  // namespace ir
}  // namespace mrt
