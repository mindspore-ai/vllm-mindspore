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

#ifndef __PARSER_AST_VISITOR_H__
#define __PARSER_AST_VISITOR_H__

#include <vector>

#include "common/logger.h"
#include "parser/ast_node.h"

namespace parser {
class NodeVisitor {
public:
  virtual void Visit(StmtConstPtr stmt) {
    if (stmt == nullptr) {
      LOG_ERROR << "Null stmt node.";
    } else if (stmt->type == StmtType_End) {
      LOG_ERROR << "Invalid stmt node.";
    } else if (stmt->type == StmtType_Return) {
      Visit(stmt->stmt.Return.value);
    } else if (stmt->type == StmtType_Assign) {
      Visit(stmt->stmt.Assign.target);
      Visit(stmt->stmt.Assign.value);
    } else if (stmt->type == StmtType_AugAssign) {
      Visit(stmt->stmt.AugAssign.target);
      Visit(stmt->stmt.AugAssign.value);
    } else if (stmt->type == StmtType_Function) {
      Visit(stmt->stmt.Function.name);
      VisitList(stmt->stmt.Function.argsLen, stmt->stmt.Function.args);
      VisitList(stmt->stmt.Function.len, stmt->stmt.Function.body);
    } else if (stmt->type == StmtType_Class) {
      Visit(stmt->stmt.Class.name);
      Visit(stmt->stmt.Class.bases);
      VisitList(stmt->stmt.Class.len, stmt->stmt.Class.body);
    } else if (stmt->type == StmtType_Block) {
      VisitList(stmt->stmt.Block.len, stmt->stmt.Block.body);
    } else if (stmt->type == StmtType_StdCin) {
      Visit(stmt->stmt.StdCin.value);
    } else if (stmt->type == StmtType_StdCout) {
      Visit(stmt->stmt.StdCout.value);
    } else if (stmt->type == StmtType_If) {
      Visit(stmt->stmt.If.condition);
      VisitList(stmt->stmt.If.ifLen, stmt->stmt.If.ifBody);
      VisitList(stmt->stmt.If.elseLen, stmt->stmt.If.elseBody);
    } else if (stmt->type == StmtType_For) {
      Visit(stmt->stmt.For.element);
      Visit(stmt->stmt.For.iterator);
      VisitList(stmt->stmt.For.len, stmt->stmt.For.body);
    } else if (stmt->type == StmtType_While) {
      Visit(stmt->stmt.While.condition);
      VisitList(stmt->stmt.While.len, stmt->stmt.While.body);
    } else if (stmt->type == StmtType_Expr) {
      Visit(stmt->stmt.Expr.value);
    } else if (stmt->type == StmtType_Module) {
      VisitList(stmt->stmt.Module.len, stmt->stmt.Module.body);
    }
  }

  virtual void Visit(ExprConstPtr expr) {
    if (expr == nullptr) {
      LOG_ERROR << "Null expr node.";
    } else if (expr->type == ExprType_End) {
      LOG_ERROR << "Invalid expr node.";
    } else if (expr->type == ExprType_Binary) {
      Visit(expr->expr.Binary.left);
      Visit(expr->expr.Binary.right);
    } else if (expr->type == ExprType_Unary) {
      Visit(expr->expr.Unary.operand);
    } else if (expr->type == ExprType_Name) {
      // No expr.
    } else if (expr->type == ExprType_Literal) {
      // No expr.
    } else if (expr->type == ExprType_List) {
      VisitList(expr->expr.List.len, expr->expr.List.values);
    } else if (expr->type == ExprType_Call) {
      Visit(expr->expr.Call.function);
      Visit(expr->expr.Call.list);
    } else if (expr->type == ExprType_Attribute) {
      Visit(expr->expr.Attribute.entity);
      Visit(expr->expr.Attribute.attribute);
    } else {
      // No expr.
    }
  }

  virtual void VisitList(size_t len, StmtConstPtr *stmtPtr) {
    for (size_t i = 0; i < len; ++i) {
      Visit(stmtPtr[i]);
    }
  }

  virtual void VisitList(size_t len, ExprConstPtr *exprPtr) {
    for (size_t i = 0; i < len; ++i) {
      Visit(exprPtr[i]);
    }
  }

  virtual ~NodeVisitor() {}
};
} // namespace parser

#endif // __PARSER_AST_VISITOR_H__