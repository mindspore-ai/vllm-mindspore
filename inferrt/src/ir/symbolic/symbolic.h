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

#ifndef __IR_SYMBOLIC_SYMBOLIC_H__
#define __IR_SYMBOLIC_SYMBOLIC_H__

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include "ir/common/intrusive_ptr.h"

namespace mrt {
namespace ir {

class SymbolicExpr;
using SymbolicExprPtr = IntrusivePtr<SymbolicExpr>;

class SymbolicExpr : public RefCounted {
 public:
  enum class Kind {
    Constant,
    Symbol,
    Add,
    Mul,
    TrueDiv,
    FloorDiv,
    CeilDiv,
  };

  explicit SymbolicExpr(Kind kind) : kind_(kind) {}
  virtual ~SymbolicExpr() = default;
  Kind GetKind() const { return kind_; }
  virtual int64_t Evaluate() const = 0;
  virtual std::string ToString() const = 0;

 private:
  Kind kind_;
};

class SymbolicConst : public SymbolicExpr {
 public:
  explicit SymbolicConst(int64_t value) : SymbolicExpr(Kind::Constant), value_(value) {}
  int64_t Evaluate() const override { return value_; }
  std::string ToString() const override { return std::to_string(value_); }
  int64_t GetValue() const { return value_; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::Constant; }

 private:
  int64_t value_;
};

class SymbolicVar : public SymbolicExpr {
 public:
  explicit SymbolicVar(const std::string &name) : SymbolicExpr(Kind::Symbol), name_(name), value_(-1) {}
  int64_t Evaluate() const override;
  std::string ToString() const override { return name_; }
  void SetValue(int64_t value) { value_ = value; }
  const std::string &GetName() const { return name_; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::Symbol; }

 private:
  std::string name_;
  int64_t value_;  // for evaluation
};

class SymbolicBinaryOp : public SymbolicExpr {
 public:
  SymbolicBinaryOp(Kind kind, SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicExpr(kind), lhs_(lhs), rhs_(rhs) {}

  SymbolicExprPtr getLHS() const { return lhs_; }
  SymbolicExprPtr getRHS() const { return rhs_; }

 protected:
  SymbolicExprPtr lhs_;
  SymbolicExprPtr rhs_;
};

class SymbolicAdd : public SymbolicBinaryOp {
 public:
  SymbolicAdd(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Add, lhs, rhs) {}
  int64_t Evaluate() const override { return lhs_->Evaluate() + rhs_->Evaluate(); }
  std::string ToString() const override { return "(" + lhs_->ToString() + " + " + rhs_->ToString() + ")"; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::Add; }
};

class SymbolicMul : public SymbolicBinaryOp {
 public:
  SymbolicMul(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::Mul, lhs, rhs) {}
  int64_t Evaluate() const override { return lhs_->Evaluate() * rhs_->Evaluate(); }
  std::string ToString() const override { return "(" + lhs_->ToString() + " * " + rhs_->ToString() + ")"; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::Mul; }
};

class SymbolicTrueDiv : public SymbolicBinaryOp {
 public:
  SymbolicTrueDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::TrueDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override { return "(" + lhs_->ToString() + " / " + rhs_->ToString() + ")"; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::TrueDiv; }
};

class SymbolicFloorDiv : public SymbolicBinaryOp {
 public:
  SymbolicFloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::FloorDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override { return "floor_div(" + lhs_->ToString() + ", " + rhs_->ToString() + ")"; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::FloorDiv; }
};

class SymbolicCeilDiv : public SymbolicBinaryOp {
 public:
  SymbolicCeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs) : SymbolicBinaryOp(Kind::CeilDiv, lhs, rhs) {}
  int64_t Evaluate() const override;
  std::string ToString() const override { return "ceil_div(" + lhs_->ToString() + ", " + rhs_->ToString() + ")"; }

  static bool classof(const SymbolicExpr *e) { return e->GetKind() == Kind::CeilDiv; }
};

// A helper to create symbolic expressions
SymbolicExprPtr operator+(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr operator*(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr operator/(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr FloorDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs);
SymbolicExprPtr CeilDiv(SymbolicExprPtr lhs, SymbolicExprPtr rhs);

}  // namespace ir
}  // namespace mrt

#endif  // __IR_SYMBOLIC_SYMBOLIC_H__
