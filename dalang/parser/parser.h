#ifndef __PARSER_PARSER_H__
#define __PARSER_PARSER_H__

#include <string>

#include "lexer/lexer.h"
#include "parser/ast_node.h"

namespace parser {
using namespace lexer;
class Parser {
public:
  explicit Parser(const std::string &filename);
  ~Parser() {
    ClearExprPool();
    ClearStmtPool();
  }

  // Parse expression.
  ExprPtr ParseExpr();
  ExprPtr ParseLogicalOr();
  ExprPtr ParseLogicalAnd();
  ExprPtr ParseLogical();
  ExprPtr ParseComparison();
  ExprPtr ParseAdditive();
  ExprPtr ParseMultiplicative();
  ExprPtr ParseUnary();
  ExprPtr ParseAttribute(ExprPtr entity);
  ExprPtr ParseCall(ExprPtr func);
  ExprPtr ParseCallAndAttribute();
  ExprPtr ParseGroup();
  ExprPtr ParsePrimary();
  ExprPtr ParseIdentifier();
  ExprPtr ParseLiteral();

  // Parse statement.
  StmtPtr ParseStmtExpr();
  StmtPtr ParseAssign();
  StmtPtr ParseReturn();
  StmtPtr ParserFunctionDef();
  StmtPtr ParserClassDef();
  StmtPtr ParseIf();
  StmtPtr ParseFor();
  StmtPtr ParseWhile();
  StmtPtr ParserBlock();

  // Parse statements.
  StmtsPtr ParseCode();

  void DumpAst();

private:
  TokenConstPtr PreviousToken() {
    if (tokenPos_ - 1 >= lexer_.Tokens().size()) {
      return nullptr;
    }
    return &lexer_.Tokens()[tokenPos_ - 1];
  }
  TokenConstPtr NextToken() {
    if (tokenPos_ + 1 >= lexer_.Tokens().size()) {
      return nullptr;
    }
    return &lexer_.Tokens()[tokenPos_ + 1];
  }
  TokenConstPtr CurrentToken() {
    if (tokenPos_ >= lexer_.Tokens().size()) {
      return nullptr;
    }
    return &lexer_.Tokens()[tokenPos_];
  }
  TokenConstPtr GetToken() {
    if (tokenPos_ >= lexer_.Tokens().size()) {
      CompileMessage(LineString(&lexer_.Tokens().back()),
                     "warning: tokens were exhaused");
      exit(1);
    }
    const auto *token = &lexer_.Tokens()[tokenPos_];
    ++tokenPos_;
    return token;
  }
  void RemoveToken() {
    if (tokenPos_ >= lexer_.Tokens().size()) {
      CompileMessage(LineString(&lexer_.Tokens().back()),
                     "warning: tokens were exhaused");
      exit(1);
    }
    ++tokenPos_;
  }

  size_t TokenPos() { return tokenPos_; }
  void SetTokenPos(size_t pos) { tokenPos_ = pos; }

  bool Finish() { return tokenPos_ == lexer_.Tokens().size(); }

  bool ParseStmts(StmtsPtr stmts);

  const std::string LineString(TokenConstPtr token);
  const std::string LineString();
  const std::string LineString(ExprConstPtr expr);
  const std::string LineString(StmtConstPtr stmt);

  Lexer lexer_;
  std::string filename_;
  Stmts stmts_;
  size_t tokenPos_{0};
};
} // namespace parser

#endif // __PARSER_PARSER_H__