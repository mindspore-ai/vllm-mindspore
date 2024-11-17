#ifndef __PARSER_STMT_H__
#define __PARSER_STMT_H__

#include <vector>

#include "common/common.h"
#include "lexer/token.h"
#include "parser/ast_node.h"

using namespace lexer;

namespace parser {
namespace StmtPattern {
namespace ExpressionPattern {
static inline bool Match(TokenConstPtr token) { return true; }
} // namespace ExpressionPattern

namespace AssignPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator && token->data.op == OpId_Assign) {
    return true;
  }
  return false;
}
} // namespace AssignPattern

namespace ReturnPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_return) {
    return true;
  }
  return false;
}
} // namespace ReturnPattern

namespace FunctionPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_function) {
    return true;
  }
  return false;
}

static inline bool MatchBodyStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_LeftBrace) {
    return true;
  }
  return false;
}

static inline bool MatchBodyEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_RightBrace) {
    return true;
  }
  return false;
}
} // namespace FunctionPattern

namespace ClassPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_class) {
    return true;
  }
  return false;
}

static inline bool MatchBodyStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_LeftBrace) {
    return true;
  }
  return false;
}

static inline bool MatchBodyEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_RightBrace) {
    return true;
  }
  return false;
}
} // namespace ClassPattern

namespace IfPattern {
static inline bool MatchIf(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_if) {
    return true;
  }
  return false;
}

static inline bool MatchElse(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_else) {
    return true;
  }
  return false;
}

static inline bool MatchBodyStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_LeftBrace) {
    return true;
  }
  return false;
}

static inline bool MatchBodyEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_RightBrace) {
    return true;
  }
  return false;
}
} // namespace IfPattern

namespace ForPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_for) {
    return true;
  }
  return false;
}

static inline bool MatchBodyStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_LeftBrace) {
    return true;
  }
  return false;
}

static inline bool MatchBodyEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_RightBrace) {
    return true;
  }
  return false;
}
} // namespace ForPattern

namespace WhilePattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Keyword && token->data.kw == KwId_while) {
    return true;
  }
  return false;
}

static inline bool MatchBodyStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_LeftBrace) {
    return true;
  }
  return false;
}

static inline bool MatchBodyEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_RightBrace) {
    return true;
  }
  return false;
}
} // namespace WhilePattern
} // namespace StmtPattern

#ifdef DEBUG
#define RETURN_AND_TRACE_STMT_NODE(stmt)                                       \
  LOG_OUT << "TRACE STMT: " << ToString(stmt) << ", lineno: [("                \
          << stmt->lineStart << ',' << stmt->columnStart << ")~("              \
          << stmt->lineEnd << ',' << stmt->columnEnd << "))" << LOG_ENDL;      \
  return stmt
#else
#define RETURN_AND_TRACE_STMT_NODE(stmt) return stmt
#endif

static inline StmtPtr MakeExprStmt(ExprConstPtr expr) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Expr;
  stmt->stmt.Expr.value = expr;
  stmt->lineStart = expr->lineStart;
  stmt->lineEnd = expr->lineEnd;
  stmt->columnStart = expr->columnStart;
  stmt->columnEnd = expr->columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
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
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeReturnStmt(ExprConstPtr value) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Return;
  stmt->stmt.Return.value = value;
  stmt->lineStart = value->lineStart;
  stmt->lineEnd = value->lineEnd;
  stmt->columnStart = value->columnStart;
  stmt->columnEnd = value->columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeFunctionStmt(ExprConstPtr id, ExprConstPtr args,
                                       Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Function;
  stmt->stmt.Function.name = id;
  stmt->stmt.Function.args = args;
  stmt->stmt.Function.len = body.size();
  stmt->stmt.Function.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.Function.body[i] = body[i];
  }
  stmt->lineStart = id->lineStart;
  stmt->lineEnd = body.back()->lineEnd;
  stmt->columnStart = id->columnStart;
  stmt->columnEnd = body.back()->columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeClassStmt(ExprConstPtr id, ExprConstPtr bases,
                                    Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Class;
  stmt->stmt.Class.name = id;
  stmt->stmt.Class.bases = bases;
  stmt->stmt.Class.len = body.size();
  stmt->stmt.Class.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.Class.body[i] = body[i];
  }
  stmt->lineStart = id->lineStart;
  stmt->lineEnd = body.back()->lineEnd;
  stmt->columnStart = id->columnStart;
  stmt->columnEnd = body.back()->columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeIfStmt(ExprConstPtr cond, Stmts &ifBody,
                                 Stmts &elseBody) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_If;
  stmt->stmt.If.condition = cond;
  stmt->stmt.If.ifLen = ifBody.size();
  stmt->stmt.If.ifBody =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * ifBody.size());
  for (size_t i = 0; i < ifBody.size(); ++i) {
    stmt->stmt.If.ifBody[i] = ifBody[i];
  }
  stmt->stmt.If.elseLen = elseBody.size();
  stmt->stmt.If.elseBody =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * elseBody.size());
  for (size_t i = 0; i < elseBody.size(); ++i) {
    stmt->stmt.If.elseBody[i] = elseBody[i];
  }
  stmt->lineStart = cond->lineStart;
  int lineEnd;
  if (!elseBody.empty()) {
    lineEnd = elseBody.back()->lineEnd;
  } else if (!ifBody.empty()) {
    lineEnd = ifBody.back()->lineEnd;
  } else {
    lineEnd = cond->lineEnd;
  }
  stmt->lineEnd = lineEnd;
  stmt->columnStart = cond->columnStart;
  int columnEnd;
  if (!elseBody.empty()) {
    columnEnd = elseBody.back()->columnEnd;
  } else if (!ifBody.empty()) {
    columnEnd = ifBody.back()->columnEnd;
  } else {
    columnEnd = cond->columnEnd;
  }
  stmt->columnEnd = columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}
} // namespace parser

#endif // __PARSER_STMT_H__