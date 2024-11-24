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

namespace AugAssignPattern {
static inline bool Match(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Operator &&
      (token->data.op == OpId_AddAssign || token->data.op == OpId_SubAssign ||
       token->data.op == OpId_MulAssign || token->data.op == OpId_DivAssign ||
       token->data.op == OpId_ModAssign)) {
    return true;
  }
  return false;
}
} // namespace AugAssignPattern

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

static inline bool MatchArgsStart(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_LeftParenthesis) {
    return true;
  }
  return false;
}

static inline bool MatchArgsSeparator(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_Comma) {
    return true;
  }
  return false;
}

static inline bool MatchArgsEnd(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator &&
      token->data.sp == SpId_RightParenthesis) {
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

namespace BlockPattern {
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
} // namespace BlockPattern

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

static inline bool MatchIteratorSeparator(TokenConstPtr token) {
  if (token == nullptr) {
    return false;
  }
  if (token->type == TokenType_Separator && token->data.sp == SpId_Colon) {
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
          << stmt->lineEnd << ',' << stmt->columnEnd << "))";                  \
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

static inline StmtPtr MakeAugAssignStmt(ExprConstPtr target, OpId op,
                                        ExprConstPtr value) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_AugAssign;
  stmt->stmt.AugAssign.target = target;
  if (op == OpId_AddAssign) {
    stmt->stmt.AugAssign.op = OpId_Add;
  } else if (op == OpId_SubAssign) {
    stmt->stmt.AugAssign.op = OpId_Sub;
  } else if (op == OpId_MulAssign) {
    stmt->stmt.AugAssign.op = OpId_Mul;
  } else if (op == OpId_DivAssign) {
    stmt->stmt.AugAssign.op = OpId_Div;
  } else if (op == OpId_ModAssign) {
    stmt->stmt.AugAssign.op = OpId_Mod;
  } else {
    return nullptr;
  }
  stmt->stmt.AugAssign.value = value;
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
  if (value != nullptr) {
    stmt->lineStart = value->lineStart;
    stmt->lineEnd = value->lineEnd;
    stmt->columnStart = value->columnStart;
    stmt->columnEnd = value->columnEnd;
  }
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeFunctionStmt(ExprConstPtr id, const Stmts &args,
                                       const Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Function;
  stmt->stmt.Function.name = id;
  stmt->stmt.Function.argsLen = args.size();
  stmt->stmt.Function.args =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * args.size());
  for (size_t i = 0; i < args.size(); ++i) {
    stmt->stmt.Function.args[i] = args[i];
  }
  stmt->stmt.Function.len = body.size();
  stmt->stmt.Function.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.Function.body[i] = body[i];
  }
  stmt->lineStart = id->lineStart;
  if (!body.empty()) {
    stmt->lineEnd = body.back()->lineEnd;
  }
  stmt->columnStart = id->columnStart;
  if (!body.empty()) {
    stmt->columnEnd = body.back()->columnEnd;
  }
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeClassStmt(ExprConstPtr id, ExprConstPtr bases,
                                    const Stmts &body) {
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
  int lineEnd;
  if (body.empty()) {
    lineEnd = id->lineEnd;
  } else {
    lineEnd = body.back()->lineEnd;
  }
  stmt->lineEnd = lineEnd;
  stmt->columnStart = id->columnStart;
  int columnEnd;
  if (body.empty()) {
    columnEnd = id->columnEnd;
  } else {
    columnEnd = body.back()->columnEnd;
  }
  stmt->columnEnd = columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeBlockStmt(const Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Block;
  stmt->stmt.Block.len = body.size();
  stmt->stmt.Block.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.Block.body[i] = body[i];
  }

  if (!body.empty()) {
    stmt->lineStart = body.front()->lineStart;
    stmt->lineEnd = body.back()->lineEnd;
    stmt->columnStart = body.front()->columnStart;
    stmt->columnEnd = body.back()->columnEnd;
  }
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeIfStmt(ExprConstPtr cond, const Stmts &ifBody,
                                 const Stmts &elseBody) {
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

static inline StmtPtr MakeForStmt(ExprConstPtr elem, ExprConstPtr iter,
                                  const Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_For;
  stmt->stmt.For.element = elem;
  stmt->stmt.For.iterator = iter;
  stmt->stmt.For.len = body.size();
  stmt->stmt.For.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.For.body[i] = body[i];
  }
  stmt->lineStart = elem->lineStart;
  int lineEnd;
  if (!body.empty()) {
    lineEnd = body.back()->lineEnd;
  } else {
    lineEnd = iter->lineEnd;
  }
  stmt->lineEnd = lineEnd;
  stmt->columnStart = elem->columnStart;
  int columnEnd;
  if (!body.empty()) {
    columnEnd = body.back()->columnEnd;
  } else {
    columnEnd = iter->columnEnd;
  }
  stmt->columnEnd = columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeWhileStmt(ExprConstPtr cond, const Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_While;
  stmt->stmt.While.condition = cond;
  stmt->stmt.While.len = body.size();
  stmt->stmt.While.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.While.body[i] = body[i];
  }
  stmt->lineStart = cond->lineStart;
  int lineEnd;
  if (!body.empty()) {
    lineEnd = body.back()->lineEnd;
  } else {
    lineEnd = cond->lineEnd;
  }
  stmt->lineEnd = lineEnd;
  stmt->columnStart = cond->columnStart;
  int columnEnd;
  if (!body.empty()) {
    columnEnd = body.back()->columnEnd;
  } else {
    columnEnd = cond->columnEnd;
  }
  stmt->columnEnd = columnEnd;
  RETURN_AND_TRACE_STMT_NODE(stmt);
}

static inline StmtPtr MakeModuleStmt(const Stmts &body) {
  StmtPtr stmt = NewStmt();
  stmt->type = StmtType_Module;
  stmt->stmt.Module.len = body.size();
  stmt->stmt.Module.body =
      (StmtConstPtr *)malloc(sizeof(StmtConstPtr) * body.size());
  for (size_t i = 0; i < body.size(); ++i) {
    stmt->stmt.Module.body[i] = body[i];
  }
  if (!body.empty()) {
    stmt->lineStart = body.front()->lineStart;
    stmt->lineEnd = body.back()->lineEnd;
    stmt->columnStart = body.front()->columnStart;
    stmt->columnEnd = body.back()->columnEnd;
  }
  RETURN_AND_TRACE_STMT_NODE(stmt);
}
} // namespace parser

#endif // __PARSER_STMT_H__