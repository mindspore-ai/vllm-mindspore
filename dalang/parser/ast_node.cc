#include "parser/ast_node.h"

#include <vector>

namespace parser {
// Expr pool.
static Exprs gExprPool;

ExprPtr NewExpr() {
  (void)gExprPool.emplace_back((ExprPtr)malloc(sizeof(Expr)));
  return gExprPool.back();
}

void ClearExprListMemory(ExprPtr expr) {
  if (expr->type == ExprType_List && expr->expr.List.len != 0 &&
      expr->expr.List.values != nullptr) {
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

void ClearStmtFunctionBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Function && stmt->stmt.Function.len != 0 &&
      stmt->stmt.Function.body != nullptr) {
    free(stmt->stmt.Function.body);
    stmt->stmt.Function.body = nullptr;
    stmt->stmt.Function.len = 0;
  }
}

void ClearStmtClassBodyMemory(StmtPtr stmt) {
  if (stmt->type == StmtType_Class && stmt->stmt.Class.len != 0 &&
      stmt->stmt.Class.body != nullptr) {
    free(stmt->stmt.Class.body);
    stmt->stmt.Class.body = nullptr;
    stmt->stmt.Class.len = 0;
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

void ClearStmtPool() {
  for (StmtPtr stmt : gStmtPool) {
    ClearStmtFunctionBodyMemory(stmt);
    ClearStmtClassBodyMemory(stmt);
    ClearStmtIfMemory(stmt);
    free(stmt);
  }
  gStmtPool.clear();
}

StmtsConstPtr StmtList() { return &gStmtPool; }

const std::string ToString(StmtConstPtr stmt) {
  if (stmt->type == StmtType_Return) {
    return "Return";
  } else if (stmt->type == StmtType_Assign) {
    return "Assign";
  } else if (stmt->type == StmtType_Function) {
    return "Function";
  } else if (stmt->type == StmtType_Class) {
    return "Class";
  } else if (stmt->type == StmtType_If) {
    return "If";
  } else if (stmt->type == StmtType_Expr) {
    return "Expr";
  }
  return "...";
}

const std::string ToString(ExprConstPtr expr) {
  if (expr == nullptr) {
    return "Expr{null}";
  }
  if (expr->type == ExprType_Binary) {
    return "Binary{" + std::string(ToStr(expr->expr.Binary.op)) + "}";
  } else if (expr->type == ExprType_Unary) {
    return "Unary{" + std::string(ToStr(expr->expr.Unary.op)) + "}";
  } else if (expr->type == ExprType_Name) {
    return "Name{" + *expr->expr.Name.name + "}";
  } else if (expr->type == ExprType_Literal) {
    return "Literal{" + std::string(ToStr(expr->expr.Literal.kind)) + "/" +
           *expr->expr.Literal.value + "}";
  } else if (expr->type == ExprType_List) {
    return "List{len:" + std::to_string(expr->expr.List.len) + "}";
  } else if (expr->type == ExprType_Call) {
    return "Call";
  } else if (expr->type == ExprType_Attribute) {
    return "Attribute";
  }
  return "?";
}
} // namespace parser
