#ifndef __PARSER_AST_VISITOR_H__
#define __PARSER_AST_VISITOR_H__

#include <vector>

#include "common/common.h"
#include "parser/ast_node.h"

namespace parser {
class NodeVisitor {
public:
  virtual void Visit(StmtsConstPtr stmts) {
    if (stmts == nullptr) {
      LOG_ERROR << "Null stmts node." << LOG_ENDL;
      return;
    }
    for (StmtConstPtr stmt : *stmts) {
      Visit(stmt);
    }
  }

  virtual void Visit(StmtConstPtr stmt) {
    if (stmt == nullptr) {
      LOG_ERROR << "Null stmt node." << LOG_ENDL;
    } else if (stmt->type == StmtType_End) {
      LOG_ERROR << "Invalid stmt node." << LOG_ENDL;
    } else if (stmt->type == StmtType_Return) {
      Visit(stmt->stmt.Return.value);
    } else if (stmt->type == StmtType_Assign) {
      Visit(stmt->stmt.Assign.target);
      Visit(stmt->stmt.Assign.value);
    } else if (stmt->type == StmtType_Function) {
      Visit(stmt->stmt.Function.name);
      Visit(stmt->stmt.Function.args);
      VisitList(stmt->stmt.Function.len, stmt->stmt.Function.body);
    } else if (stmt->type == StmtType_Class) {
      Visit(stmt->stmt.Class.name);
      Visit(stmt->stmt.Class.bases);
      VisitList(stmt->stmt.Class.len, stmt->stmt.Class.body);
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
    }
  }

  virtual void Visit(ExprConstPtr expr) {
    if (expr == nullptr) {
      LOG_ERROR << "Null expr node." << LOG_ENDL;
    } else if (expr->type == ExprType_End) {
      LOG_ERROR << "Invalid expr node." << LOG_ENDL;
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
};
} // namespace parser

#endif // __PARSER_AST_VISITOR_H__