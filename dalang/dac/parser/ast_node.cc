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

#include "parser/ast_node.h"

#include <vector>

#include "common/common.h"

namespace da {
namespace parser {
// Expr pool.
static Exprs gExprPool;

ExprPtr NewExpr() {
  (void)gExprPool.emplace_back((ExprPtr)malloc(sizeof(Expr)));
  return gExprPool.back();
}

void ClearExprListMemory(ExprPtr expr) {
  if (expr->type == ExprType_List && expr->expr.List.len != 0 && expr->expr.List.values != nullptr) {
    free(expr->expr.List.values);
    expr->expr.List.values = nullptr;
    expr->expr.List.len = 0;
  }
}

void ClearExprPool() {
  for (ExprPtr expr : gExprPool) {
    ClearExprListMemory(expr);
    free(expr);
  }
  gExprPool.clear();
}

ExprsConstPtr ExprList() { return &gExprPool; }

// Stmt pool.
static Stmts gStmtPool;

StmtPtr NewStmt() {
  (void)gStmtPool.emplace_back((StmtPtr)malloc(sizeof(Stmt)));
  return gStmtPool.back();
}

void ClearStmtGraphArgsMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Graph && stmt->stmt.Graph.argsLen != 0 && stmt->stmt.Graph.args != nullptr) {
    free(stmt->stmt.Graph.args);
    stmt->stmt.Graph.args = nullptr;
    stmt->stmt.Graph.argsLen = 0;
  }
}

void ClearStmtGraphBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Graph && stmt->stmt.Graph.len != 0 && stmt->stmt.Graph.body != nullptr) {
    free(stmt->stmt.Graph.body);
    stmt->stmt.Graph.body = nullptr;
    stmt->stmt.Graph.len = 0;
  }
}

void ClearStmtFunctionArgsMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Function && stmt->stmt.Function.argsLen != 0 && stmt->stmt.Function.args != nullptr) {
    free(stmt->stmt.Function.args);
    stmt->stmt.Function.args = nullptr;
    stmt->stmt.Function.argsLen = 0;
  }
}

void ClearStmtFunctionBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Function && stmt->stmt.Function.len != 0 && stmt->stmt.Function.body != nullptr) {
    free(stmt->stmt.Function.body);
    stmt->stmt.Function.body = nullptr;
    stmt->stmt.Function.len = 0;
  }
}

void ClearStmtClassBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Class && stmt->stmt.Class.len != 0 && stmt->stmt.Class.body != nullptr) {
    free(stmt->stmt.Class.body);
    stmt->stmt.Class.body = nullptr;
    stmt->stmt.Class.len = 0;
  }
}

void ClearStmtBlockBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Block && stmt->stmt.Block.len != 0 && stmt->stmt.Block.body != nullptr) {
    free(stmt->stmt.Block.body);
    stmt->stmt.Block.body = nullptr;
    stmt->stmt.Block.len = 0;
  }
}

void ClearStmtIfMemory(StmtPtr stmt) {
  if (stmt->type != StmtType_If) {
    return;
  }
  if (stmt->stmt.If.ifLen != 0 && stmt->stmt.If.ifBody != nullptr) {
    free(stmt->stmt.If.ifBody);
    stmt->stmt.If.ifBody = nullptr;
    stmt->stmt.If.ifLen = 0;
  }
  if (stmt->stmt.If.elseLen != 0 && stmt->stmt.If.elseBody != nullptr) {
    free(stmt->stmt.If.elseBody);
    stmt->stmt.If.elseBody = nullptr;
    stmt->stmt.If.elseLen = 0;
  }
}

void ClearStmtForMemory(StmtPtr stmt) {
  if (stmt->type != StmtType_For) {
    return;
  }
  if (stmt->stmt.For.len != 0 && stmt->stmt.For.body != nullptr) {
    free(stmt->stmt.For.body);
    stmt->stmt.For.body = nullptr;
    stmt->stmt.For.len = 0;
  }
}

void ClearStmtWhileMemory(StmtPtr stmt) {
  if (stmt->type != StmtType_While) {
    return;
  }
  if (stmt->stmt.While.len != 0 && stmt->stmt.While.body != nullptr) {
    free(stmt->stmt.While.body);
    stmt->stmt.While.body = nullptr;
    stmt->stmt.While.len = 0;
  }
}

void ClearStmtModuleMemory(StmtPtr stmt) {
  if (stmt->type != StmtType_Module) {
    return;
  }
  if (stmt->stmt.Module.len != 0 && stmt->stmt.Module.body != nullptr) {
    free(stmt->stmt.Module.body);
    stmt->stmt.Module.body = nullptr;
    stmt->stmt.Module.len = 0;
  }
}

void ClearStmtPool() {
  for (StmtPtr stmt : gStmtPool) {
    ClearStmtGraphArgsMemory(stmt);
    ClearStmtGraphBodyMemory(stmt);
    ClearStmtFunctionArgsMemory(stmt);
    ClearStmtFunctionBodyMemory(stmt);
    ClearStmtClassBodyMemory(stmt);
    ClearStmtBlockBodyMemory(stmt);
    ClearStmtIfMemory(stmt);
    ClearStmtForMemory(stmt);
    ClearStmtWhileMemory(stmt);
    free(stmt);
  }
  gStmtPool.clear();
}

StmtsConstPtr StmtList() { return &gStmtPool; }

const std::string ToString(StmtConstPtr stmt) {
  switch (stmt->type) {
    case StmtType_Return: {
      return "Return";
    }
    case StmtType_Assign: {
      return "Assign";
    }
    case StmtType_AugAssign: {
      return "AugAssign{" + std::string(ToStr(stmt->stmt.AugAssign.op)) + '}';
    }
    case StmtType_Graph: {
      return "Graph";
    }
    case StmtType_Function: {
      return "Function";
    }
    case StmtType_Class: {
      return "Class";
    }
    case StmtType_Block: {
      return "Block";
    }
    case StmtType_StdCin: {
      return "StdCin";
    }
    case StmtType_StdCout: {
      return "StdCout";
    }
    case StmtType_If: {
      return "If";
    }
    case StmtType_For: {
      return "For";
    }
    case StmtType_While: {
      return "While";
    }
    case StmtType_Expr: {
      return "Expr";
    }
    case StmtType_Module: {
      return "Module";
    }
    default:
      return "...";
  }
}

const std::string ToString(ExprConstPtr expr) {
  if (expr == nullptr) {
    return "Expr{null}";
  }

  switch (expr->type) {
    case ExprType_Binary: {
      return std::string(ToStr(expr->expr.Binary.op));
    }
    case ExprType_Unary: {
      return std::string(ToStr(expr->expr.Unary.op));
    }
    case ExprType_Name: {
      return "Name{" + *expr->expr.Name.identifier + "}";
    }
    case ExprType_Literal: {
      return "Literal{" + std::string(ToStr(expr->expr.Literal.kind)) + ':' +
             ConvertEscapeString(*expr->expr.Literal.value) + '}';
    }
    case ExprType_List: {
      return "List{len:" + std::to_string(expr->expr.List.len) + "}";
    }
    case ExprType_Call: {
      return "Call";
    }
    case ExprType_Attribute: {
      return "Attribute";
    }
    default:
      return "?";
  }
}
}  // namespace parser
}  // namespace da
