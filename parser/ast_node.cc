#include "parser/ast_node.h"

#include <vector>

namespace parser {
// Expr pool.
static std::vector<ExprPtr> gExprPool;

ExprPtr NewExpr() {
  (void)gExprPool.emplace_back((ExprPtr)malloc(sizeof(Expr)));
  return gExprPool.back();
}

void ClearExprPool() {
  for (ExprPtr expr : gExprPool) {
    free(expr);
  }
  gExprPool.clear();
}

const std::vector<ExprPtr> &ExprList() { return gExprPool; }

// Stmt pool.
static std::vector<StmtPtr> gStmtPool;

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

const std::vector<StmtPtr> &StmtList() { return gStmtPool; }

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
  }
  return "?";
}
} // namespace parser
