#ifndef __PARSER_PARSER_H__
#define __PARSER_PARSER_H__

#include <string>

#include "lexer/lexer.h"
#include "parser/ast_node.h"

namespace parser {
using namespace lexer;
class Parser {
public:
  explicit Parser(const std::string &filename)
      : lexer_{Lexer(filename)}, filename_{filename} {}
  ~Parser() {
    ClearExprPool();
    ClearStmtPool();
  }

  // Parse expression.
  ExprPtr ParseExpr();
  ExprPtr ParseLogical();
  ExprPtr ParseComparison();
  ExprPtr ParseAdditive();
  ExprPtr ParseMultiplicative();
  ExprPtr ParseUnary();
  ExprPtr ParseAttribute();
  ExprPtr ParseCall();
  ExprPtr ParseGroup();
  ExprPtr ParsePrimary();

  // Parse statement.
  StmtPtr ParseStmt();
  StmtPtr ParseStmtExpr();
  StmtPtr ParseAssign();
  StmtPtr ParseReturn();

  // Parse statements.
  StmtsPtr ParseStmts();

  void DumpAst();

private:
  TokenConstPtr CurrentToken() { return &lexer_.Tokens()[tokenPos_]; }
  TokenConstPtr PreviousToken() { return &lexer_.Tokens()[tokenPos_ - 1]; }
  TokenConstPtr NextToken() { return &lexer_.Tokens()[tokenPos_ + 1]; }
  TokenConstPtr GetToken() {
    const auto *token = &lexer_.Tokens()[tokenPos_];
    ++tokenPos_;
    return token;
  }
  bool Finish() { return tokenPos_ == lexer_.Tokens().size(); }

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