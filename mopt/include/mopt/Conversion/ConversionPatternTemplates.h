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

#ifndef MOPT_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H
#define MOPT_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mopt {

//===----------------------------------------------------------------------===//
// Generic variadic conversion pattern template
//===----------------------------------------------------------------------===//

// Universal template for operations with arbitrary number of operands
// Usage: ConvertOp<SrcOp, DstOp, &SrcOp::Adaptor::getLhs, &SrcOp::Adaptor::getRhs>
//        ConvertOp<SrcOp, DstOp, &SrcOp::Adaptor::getSelf, &SrcOp::Adaptor::getOther, &SrcOp::Adaptor::getAlpha>
template <typename SrcOp, typename DstOp, auto... Getters>
struct ConvertOp : public mlir::OpConversionPattern<SrcOp> {
  using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();

    rewriter.replaceOpWithNewOp<DstOp>(op, resultType, (adaptor.*Getters)()...);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Utility macros for getter construction
//===----------------------------------------------------------------------===//

// Helper macro for concatenation
#define MOPT_CONCAT_IMPL(x, y) x##y
#define MOPT_CONCAT(x, y) MOPT_CONCAT_IMPL(x, y)

// Convert parameter name to getter (e.g., Lhs -> getLhs)
#define MOPT_GETTER(SrcOp, param) &SrcOp::Adaptor::MOPT_CONCAT(get, param)

//===----------------------------------------------------------------------===//
// Simplified conversion macros
//===----------------------------------------------------------------------===//

// Universal conversion macro supporting 1-5 operands
// Usage: MOPT_CONVERT(SrcOp, DstOp, Lhs, Rhs) -> getLhs(), getRhs()
//        MOPT_CONVERT(SrcOp, DstOp, Self, Other, Alpha) -> getSelf(), getOther(), getAlpha()

#define MOPT_CONVERT_1(SrcOp, DstOp, P1) ::mopt::ConvertOp<SrcOp, DstOp, MOPT_GETTER(SrcOp, P1)>

#define MOPT_CONVERT_2(SrcOp, DstOp, P1, P2) \
  ::mopt::ConvertOp<SrcOp, DstOp, MOPT_GETTER(SrcOp, P1), MOPT_GETTER(SrcOp, P2)>

#define MOPT_CONVERT_3(SrcOp, DstOp, P1, P2, P3) \
  ::mopt::ConvertOp<SrcOp, DstOp, MOPT_GETTER(SrcOp, P1), MOPT_GETTER(SrcOp, P2), MOPT_GETTER(SrcOp, P3)>

#define MOPT_CONVERT_4(SrcOp, DstOp, P1, P2, P3, P4)                                                      \
  ::mopt::ConvertOp<SrcOp, DstOp, MOPT_GETTER(SrcOp, P1), MOPT_GETTER(SrcOp, P2), MOPT_GETTER(SrcOp, P3), \
                    MOPT_GETTER(SrcOp, P4)>

#define MOPT_CONVERT_5(SrcOp, DstOp, P1, P2, P3, P4, P5)                                                  \
  ::mopt::ConvertOp<SrcOp, DstOp, MOPT_GETTER(SrcOp, P1), MOPT_GETTER(SrcOp, P2), MOPT_GETTER(SrcOp, P3), \
                    MOPT_GETTER(SrcOp, P4), MOPT_GETTER(SrcOp, P5)>

// Macro overloading based on number of arguments
#define MOPT_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, NAME, ...) NAME

#define MOPT_CONVERT(...)                                                                                     \
  MOPT_GET_MACRO(__VA_ARGS__, MOPT_CONVERT_5, MOPT_CONVERT_4, MOPT_CONVERT_3, MOPT_CONVERT_2, MOPT_CONVERT_1) \
  (__VA_ARGS__)

}  // namespace mopt

#endif  // MOPT_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H
