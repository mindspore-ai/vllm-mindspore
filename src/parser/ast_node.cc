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

void ClearStmtPool() {
  for (StmtPtr Stmt : gStmtPool) {
    free(Stmt);
  }
  gStmtPool.clear();
}

StmtsConstPtr StmtList() { return &gStmtPool; }

const std::string ToString(StmtConstPtr stmt) {
  if (stmt->type == StmtType_Return) {
    return "[return]";
  } else if (stmt->type == StmtType_Assign) {
    return "[=]";
  } else if (stmt->type == StmtType_Expr) {
    return "[Expr]";
  }
  return "...";
}

const std::string ToString(ExprConstPtr expr) {
  if (expr->type == ExprType_Binary) {
    return "Binary[" + std::string(ToStr(expr->expr.Binary.op)) + "]";
  } else if (expr->type == ExprType_Unary) {
    return "Unary[" + std::string(ToStr(expr->expr.Unary.op)) + "]";
  } else if (expr->type == ExprType_Name) {
    return "Name[" + *expr->expr.Name.name + "]";
  } else if (expr->type == ExprType_Literal) {
    return "Literal[" + std::string(ToStr(expr->expr.Literal.kind)) + "/" +
           *expr->expr.Literal.value + "]";
  } else if (expr->type == ExprType_List) {
    return "List[len:" + std::to_string(expr->expr.List.len) + "]";
  } else if (expr->type == ExprType_Call) {
    return "Call[func:" + ToString(expr->expr.Call.func) + "]";
  }
  return "?";
}
} // namespace parser
