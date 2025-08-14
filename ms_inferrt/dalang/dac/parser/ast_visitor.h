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

namespace da {
namespace parser {
class NodeVisitor {
 public:
  virtual void Visit(StmtConstPtr stmt) {
    if (stmt == nullptr) {
      LOG_ERROR << "Null stmt node.";
      return;
    }
    switch (stmt->type) {
      case StmtType_End: {
        LOG_ERROR << "Invalid stmt node.";
        break;
      }
      case StmtType_Return: {
        Visit(stmt->stmt.Return.value);
        break;
      }
      case StmtType_Assign: {
        Visit(stmt->stmt.Assign.target);
        Visit(stmt->stmt.Assign.value);
        break;
      }
      case StmtType_AugAssign: {
        Visit(stmt->stmt.AugAssign.target);
        Visit(stmt->stmt.AugAssign.value);
        break;
      }
      case StmtType_Graph: {
        Visit(stmt->stmt.Graph.name);
        VisitList(stmt->stmt.Graph.argsLen, stmt->stmt.Graph.args);
        VisitList(stmt->stmt.Graph.len, stmt->stmt.Graph.body);
        break;
      }
      case StmtType_Function: {
        Visit(stmt->stmt.Function.name);
        VisitList(stmt->stmt.Function.argsLen, stmt->stmt.Function.args);
        VisitList(stmt->stmt.Function.len, stmt->stmt.Function.body);
        break;
      }
      case StmtType_Class: {
        Visit(stmt->stmt.Class.name);
        Visit(stmt->stmt.Class.bases);
        VisitList(stmt->stmt.Class.len, stmt->stmt.Class.body);
        break;
      }
      case StmtType_Block: {
        VisitList(stmt->stmt.Block.len, stmt->stmt.Block.body);
        break;
      }
      case StmtType_StdCin: {
        Visit(stmt->stmt.StdCin.value);
        break;
      }
      case StmtType_StdCout: {
        Visit(stmt->stmt.StdCout.value);
        break;
      }
      case StmtType_If: {
        Visit(stmt->stmt.If.condition);
        VisitList(stmt->stmt.If.ifLen, stmt->stmt.If.ifBody);
        VisitList(stmt->stmt.If.elseLen, stmt->stmt.If.elseBody);
        break;
      }
      case StmtType_For: {
        Visit(stmt->stmt.For.element);
        Visit(stmt->stmt.For.iterator);
        VisitList(stmt->stmt.For.len, stmt->stmt.For.body);
        break;
      }
      case StmtType_While: {
        Visit(stmt->stmt.While.condition);
        VisitList(stmt->stmt.While.len, stmt->stmt.While.body);
        break;
      }
      case StmtType_Expr: {
        Visit(stmt->stmt.Expr.value);
        break;
      }
      case StmtType_Module: {
        VisitList(stmt->stmt.Module.len, stmt->stmt.Module.body);
        break;
      }
      default:
        break;
    }
  }

  virtual void Visit(ExprConstPtr expr) {
    if (expr == nullptr) {
      LOG_ERROR << "Null expr node.";
      return;
    }

    switch (expr->type) {
      case ExprType_End: {
        LOG_ERROR << "Invalid expr node.";
        break;
      }
      case ExprType_Binary: {
        Visit(expr->expr.Binary.left);
        Visit(expr->expr.Binary.right);
        break;
      }
      case ExprType_Unary: {
        Visit(expr->expr.Unary.operand);
        break;
      }
      case ExprType_Name: {
        // No expr.
        break;
      }
      case ExprType_Literal: {
        // No expr.
        break;
      }
      case ExprType_List: {
        VisitList(expr->expr.List.len, expr->expr.List.values);
        break;
      }
      case ExprType_Call: {
        Visit(expr->expr.Call.function);
        Visit(expr->expr.Call.list);
        break;
      }
      case ExprType_Attribute: {
        Visit(expr->expr.Attribute.entity);
        Visit(expr->expr.Attribute.attribute);
        break;
      }
      default: {
        // No expr.
        break;
      }
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
}  // namespace parser
}  // namespace da

#endif  // __PARSER_AST_VISITOR_H__