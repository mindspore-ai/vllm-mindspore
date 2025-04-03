/**
 * Copyright 2024 Zhang Qinghua
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

#ifndef __PARSER_EXPR_H__
#define __PARSER_EXPR_H__

#include <vector>

#include "common/logger.h"
#include "lexer/token.h"
#include "parser/ast_node.h"

using namespace lexer;

namespace parser {
namespace ExprPattern {
namespace LogicalPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_LogicalOr || token->data.op == OpId_LogicalAnd)) {
    return true;
  }
  return false;
}

static inline bool MatchOr(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator && token->data.op == OpId_LogicalOr) {
    return true;
  }
  return false;
}

static inline bool MatchAnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator && token->data.op == OpId_LogicalAnd) {
    return true;
  }
  return false;
}
} // namespace LogicalPattern

namespace ComparisonPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_Equal || token->data.op == OpId_GreaterEqual ||
       token->data.op == OpId_LessEqual || token->data.op == OpId_GreaterThan ||
       token->data.op == OpId_LessThan || token->data.op == OpId_NotEqual)) {
    return true;
  }
  return false;
}
} // namespace ComparisonPattern

namespace AdditivePattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_Add || token->data.op == OpId_Sub)) {
    return true;
  }
  return false;
}
} // namespace AdditivePattern

namespace MultiplicativePattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_Mul || token->data.op == OpId_Div ||
       token->data.op == OpId_Mod)) {
    return true;
  }
  return false;
}
} // namespace MultiplicativePattern

namespace UnaryPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator && token->data.op == OpId_Sub) {
    return true;
  }
  return false;
}
} // namespace UnaryPattern

namespace CallPattern {
static inline bool Match(TokenConstPtr token) { return false; }
} // namespace CallPattern

namespace AttributePattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_Dot) {
    return true;
  }
  return false;
}
} // namespace AttributePattern

namespace ListPattern {
static inline bool MatchStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_LeftParenthesis) {
    return true;
  }
  return false;
}

static inline bool MatchSplit(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_Comma) {
    return true;
  }
  return false;
}

static inline bool MatchEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_RightParenthesis) {
    return true;
  }
  return false;
}

static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (MatchStart(token) || MatchSplit(token) || MatchEnd(token)) {
    return true;
  }
  return false;
}
} // namespace ListPattern

namespace TensorPattern {
static inline bool MatchStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_LeftBracket) {
    return true;
  }
  return false;
}

static inline bool MatchSplit(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_Comma) {
    return true;
  }
  return false;
}

static inline bool MatchEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_RightBracket) {
    return true;
  }
  return false;
}

static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (MatchStart(token) || MatchSplit(token) || MatchEnd(token)) {
    return true;
  }
  return false;
}
} // namespace TensorPattern

namespace PrimaryPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword || token->type == TokenType_Identifier ||
      token->type == TokenType_Literal || token->type == TokenType_Comment) {
    return true;
  }
  return false;
}

static inline bool MatchKeyword(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword) {
    return true;
  }
  return false;
}

static inline bool MatchKeywordOps(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && (token->data.kw == KwId_ops)) {
    return true;
  }
  return false;
}

static inline bool MatchKeywordThis(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword &&
      (token->data.kw == KwId_this || token->data.kw == KwId_self)) {
    return true;
  }
  return false;
}

static inline bool MatchIdentifier(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Identifier) {
    return true;
  }
  return false;
}

static inline bool MatchLiteral(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Literal) {
    return true;
  }
  return false;
}

static inline bool MatchComment(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Comment) {
    return true;
  }
  return false;
}
} // namespace PrimaryPattern
} // namespace ExprPattern

#ifdef DEBUG
#define RETURN_AND_TRACE_EXPR_NODE(expr)                                       \
  LOG_OUT << "TRACE EXPR: " << ToString(expr) << ", lineno: [("                \
          << expr->lineStart << ',' << expr->columnStart << ")~("              \
          << expr->lineEnd << ',' << expr->columnEnd << "))";                  \
  return expr
#else
#define RETURN_AND_TRACE_EXPR_NODE(expr) return expr
#endif

static inline ExprPtr MakeBinaryExpr(TokenConstPtr op, ExprConstPtr left,
                                     ExprConstPtr right) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Binary;
  expr->expr.Binary.op = op->data.op;
  expr->expr.Binary.left = left;
  expr->expr.Binary.right = right;
  expr->lineStart = left->lineStart;
  expr->lineEnd = right->lineEnd;
  expr->columnStart = left->columnStart;
  expr->columnEnd = right->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeUnaryExpr(TokenConstPtr op, ExprConstPtr operand) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Unary;
  expr->expr.Unary.op = op->data.op;
  expr->expr.Unary.operand = operand;
  expr->lineStart = op->lineStart;
  expr->lineEnd = operand->lineEnd;
  expr->columnStart = op->columnStart;
  expr->columnEnd = operand->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeNameExpr(TokenConstPtr name) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Name;
  expr->expr.Name.identifier = &name->name;
  expr->lineStart = name->lineStart;
  expr->lineEnd = name->lineEnd;
  expr->columnStart = name->columnStart;
  expr->columnEnd = name->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeLiteralExpr(TokenConstPtr literal) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Literal;
  expr->expr.Literal.kind = literal->data.lt;
  expr->expr.Literal.value = &literal->name;
  expr->lineStart = literal->lineStart;
  expr->lineEnd = literal->lineEnd;
  expr->columnStart = literal->columnStart;
  expr->columnEnd = literal->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeTensorExpr(TokenConstPtr literal) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Literal;
  expr->expr.Literal.kind = LiteralId_tensor;
  expr->expr.Literal.value = &literal->name;
  expr->lineStart = literal->lineStart;
  expr->lineEnd = literal->lineEnd;
  expr->columnStart = literal->columnStart;
  expr->columnEnd = literal->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeListExpr(TokenConstPtr start, TokenConstPtr end,
                                   Exprs &elements) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_List;
  expr->expr.List.len = elements.size();
  expr->expr.List.values =
      (ExprConstPtr *)malloc(sizeof(ExprConstPtr) * elements.size());
  for (size_t i = 0; i < elements.size(); ++i) {
    expr->expr.List.values[i] = elements[i];
  }
  expr->lineStart = start->lineStart;
  expr->lineEnd = end->lineEnd;
  expr->columnStart = start->columnStart;
  expr->columnEnd = end->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeCallExpr(ExprConstPtr func, ExprConstPtr group) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Call;
  expr->expr.Call.function = func;
  expr->expr.Call.list = group;
  expr->lineStart = func->lineStart;
  expr->lineEnd = group->lineEnd;
  expr->columnStart = func->columnStart;
  expr->columnEnd = group->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}

static inline ExprPtr MakeAttributeExpr(ExprConstPtr entity,
                                        ExprConstPtr attribute) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Attribute;
  expr->expr.Attribute.entity = entity;
  expr->expr.Attribute.attribute = attribute;
  expr->lineStart = entity->lineStart;
  expr->lineEnd = attribute->lineEnd;
  expr->columnStart = entity->columnStart;
  expr->columnEnd = attribute->columnEnd;
  RETURN_AND_TRACE_EXPR_NODE(expr);
}
} // namespace parser

#endif // __PARSER_EXPR_H__