#ifndef __PARSER_AST_NODE_H__
#define __PARSER_AST_NODE_H__

#include <vector>

#include "common/common.h"
#include "lexer/token.h"

using namespace lexer;

namespace parser {
typedef struct ExprNode *ExprPtr;
typedef const struct ExprNode *ExprConstPtr;
typedef struct StmtNode *StmtPtr;
typedef const struct StmtNode *StmtConstPtr;
typedef std::vector<StmtPtr> Stmts;
typedef std::vector<StmtPtr> *StmtsPtr;
typedef const std::vector<StmtPtr> *StmtsConstPtr;

// Expr pool.
ExprPtr NewExpr();
void ClearExprPool();
const std::vector<ExprPtr> &ExprList();
// Stmt pool.
StmtPtr NewStmt();
void ClearStmtPool();
const std::vector<StmtPtr> &StmtList();

// Statement type.
enum StmtType {
  StmtType_Expr,
  StmtType_Assign,
  StmtType_Return,
  StmtType_Function,
  StmtType_Class,
  StmtType_If,
  StmtType_While,
  StmtType_For,
  StmtType_Break,
  StmtType_Continue,
  StmtType_Pass,
  StmtType_Import,
  StmtType_End,
};

// Statement node type.
typedef struct StmtNode {
  StmtType type = StmtType_End;
  union {
    struct {
      ExprConstPtr value;
    } Expr;
    struct {
      ExprConstPtr target;
      ExprConstPtr value;
    } Assign;
    struct {
      ExprConstPtr value;
    } Return;
  } stmt;
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
} Stmt;

namespace StmtPattern {
namespace ExpressionPattern {
static inline bool Match(TokenConstPtr token) { return true; }
} // namespace ExpressionPattern

namespace AssignPattern {
static inline bool Match(TokenConstPtr token) {
  if (token->type == TokenType_Operator && token->data.op == OpId_Assign) {
    return true;
  }
  return false;
}
} // namespace AssignPattern

namespace ReturnPattern {
static inline bool Match(TokenConstPtr token) {
  if (token->type == TokenType_Keyword && token->data.kw == KwId_return) {
    return true;
  }
  return false;
}
} // namespace ReturnPattern
} // namespace StmtPattern

enum ExprType {
  ExprType_Binary,
  ExprType_Unary,
  ExprType_Attribute,
  ExprType_Subscript,
  ExprType_List,
  ExprType_Call,
  ExprType_Name,
  ExprType_Literal,
  ExprType_End,
};

typedef struct ExprNode {
  ExprType type = ExprType_End;
  union {
    struct {
      OpId op;
      ExprConstPtr left;
      ExprConstPtr right;
    } Binary;
    struct {
      OpId op;
      ExprConstPtr operand;
    } Unary;
    struct {
      const std::string *name;
    } Name;
    struct {
      LiteralId kind;
      const std::string *value;
    } Literal;
  } expr;
  int lineStart;
  int lineEnd;
  int columnStart;
  int columnEnd;
} Expr;

namespace ExprPattern {
namespace LogicalPattern {
static inline bool Match(TokenConstPtr token) {
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_LogicalOr || token->data.op == OpId_LogicalAnd ||
       token->data.op == OpId_ShiftRight || token->data.op == OpId_ShiftLeft)) {
    return true;
  }
  return false;
}
} // namespace LogicalPattern

namespace ComparisonPattern {
static inline bool Match(TokenConstPtr token) {
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
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_Add || token->data.op == OpId_Sub)) {
    return true;
  }
  return false;
}
} // namespace AdditivePattern

namespace MultiplicativePattern {
static inline bool Match(TokenConstPtr token) {
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_Mul || token->data.op == OpId_Div)) {
    return true;
  }
  return false;
}
} // namespace MultiplicativePattern

namespace UnaryPattern {
static inline bool Match(TokenConstPtr token) {
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
static inline bool Match(TokenConstPtr token) { return false; }
} // namespace AttributePattern

namespace GroupPattern {
static inline bool Match(TokenConstPtr token) { return false; }
} // namespace GroupPattern

namespace PrimaryPattern {
static inline bool Match(TokenConstPtr token) {
  if (token->type == TokenType_Keyword || token->type == TokenType_Identifier ||
      token->type == TokenType_Literal || token->type == TokenType_Comment) {
    return true;
  }
  return false;
}

static inline bool MatchKeyword(TokenConstPtr token) {
  if (token->type == TokenType_Keyword) {
    return true;
  }
  return false;
}

static inline bool MatchIdentifier(TokenConstPtr token) {
  if (token->type == TokenType_Identifier) {
    return true;
  }
  return false;
}

static inline bool MatchLiteral(TokenConstPtr token) {
  if (token->type == TokenType_Literal) {
    return true;
  }
  return false;
}

static inline bool MatchComment(TokenConstPtr token) {
  if (token->type == TokenType_Comment) {
    return true;
  }
  return false;
}
} // namespace PrimaryPattern
} // namespace ExprPattern

static inline StmtPtr MakeExprStmt(ExprConstPtr expr) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Expr;
  stmt->stmt.Expr.value = expr;
  stmt->lineStart = expr->lineStart;
  stmt->lineEnd = expr->lineEnd;
  stmt->columnStart = expr->columnStart;
  stmt->columnEnd = expr->columnEnd;
  return stmt;
}

static inline StmtPtr MakeAssignStmt(ExprConstPtr target, ExprConstPtr value) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Assign;
  stmt->stmt.Assign.target = target;
  stmt->stmt.Assign.value = value;
  stmt->lineStart = target->lineStart;
  stmt->lineEnd = value->lineEnd;
  stmt->columnStart = target->columnStart;
  stmt->columnEnd = value->columnEnd;
  return stmt;
}

static inline StmtPtr MakeReturnStmt(ExprConstPtr value) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Return;
  stmt->stmt.Return.value = value;
  stmt->lineStart = value->lineStart;
  stmt->lineEnd = value->lineEnd;
  stmt->columnStart = value->columnStart;
  stmt->columnEnd = value->columnEnd;
  return stmt;
}

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
  return expr;
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
  return expr;
}

static inline ExprPtr MakeNameExpr(TokenConstPtr name) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Name;
  expr->expr.Name.name = &name->name;
  expr->lineStart = name->lineStart;
  expr->lineEnd = name->lineEnd;
  expr->columnStart = name->columnStart;
  expr->columnEnd = name->columnEnd;
  return expr;
}

static inline ExprPtr MakeLiteralExpr(TokenConstPtr literal) {
  ExprPtr expr = NewExpr();
  expr->type = ExprType_Literal;
  expr->expr.Literal.kind = LiteralId_int;
  expr->expr.Literal.value = &literal->name;
  expr->lineEnd = literal->lineEnd;
  expr->columnStart = literal->columnStart;
  expr->columnEnd = literal->columnEnd;
  return expr;
}

const std::string ToString(StmtConstPtr stmt);
const std::string ToString(ExprConstPtr expr);
} // namespace parser

#endif // __PARSER_AST_NODE_H__