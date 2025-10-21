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

#include "ir/symbolic/symbolic.h"
#include "common/common.h"

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
  int result = lhsVal / rhsVal;
  // If the signs of a and b are different and there's a non-zero remainder,
  // the C++ integer division rounds towards zero, which is not floor division.
  // In this case, we need to subtract 1 to round down.
  if ((lhsVal % rhsVal != 0) && ((lhsVal < 0) != (rhsVal < 0))) {
    result--;
  }
  return result;
}

SymbolicExprPtr operator+(SymbolicExprPtr lhs, SymbolicExprPtr rhs) { return MakeIntrusive<SymbolicAdd>(lhs, rhs); }

SymbolicExprPtr operator*(SymbolicExprPtr lhs, SymbolicExprPtr rhs) { return MakeIntrusive<SymbolicMul>(lhs, rhs); }

SymbolicExprPtr operator/(SymbolicExprPtr lhs, SymbolicExprPtr rhs) {
  return MakeIntrusive<SymbolicFloorDiv>(lhs, rhs);
}

}  // namespace ir
}  // namespace mrt
