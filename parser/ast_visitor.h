#ifndef __PARSER_AST_VISITOR_H__
#define __PARSER_AST_VISITOR_H__

#include <vector>

#include "common/common.h"
#include "parser/ast_node.h"

namespace parser {
class NodeVisitor {
public:
  virtual void visit(StmtsConstPtr stmts) {
    if (stmts == nullptr) {
      LOG_ERROR << "Null stmts node." << LOG_ENDL;
      return;
    }
    for (StmtConstPtr stmt : *stmts) {
      visit(stmt);
    }
  }
  virtual void visit(StmtConstPtr stmt) {
    if (stmt == nullptr) {
      LOG_ERROR << "Null stmt node." << LOG_ENDL;
    } else if (stmt->type == StmtType_End) {
      LOG_ERROR << "Invalid stmt node." << LOG_ENDL;
    } else if (stmt->type == StmtType_Return) {
      visit(stmt->stmt.Return.value);
    } else if (stmt->type == StmtType_Assign) {
      visit(stmt->stmt.Assign.target);
      visit(stmt->stmt.Assign.value);
    } else if (stmt->type == StmtType_Expr) {
      visit(stmt->stmt.Expr.value);
    }
  }
  virtual void visit(ExprConstPtr expr) {
    if (expr == nullptr) {
      LOG_ERROR << "Null expr node." << LOG_ENDL;
    } else if (expr->type == ExprType_End) {
      LOG_ERROR << "Invalid expr node." << LOG_ENDL;
    } else if (expr->type == ExprType_Binary) {
      visit(expr->expr.Binary.left);
      visit(expr->expr.Binary.right);
    } else if (expr->type == ExprType_Unary) {
      visit(expr->expr.Unary.operand);
    } else if (expr->type == ExprType_Name) {
      // No expr.
    } else if (expr->type == ExprType_Literal) {
      // No expr.
    } else {
      // No expr.
    }
  }
};
} // namespace parser

#endif // __PARSER_AST_VISITOR_H__