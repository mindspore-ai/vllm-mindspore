/**
 * Copyright 2024 Zhang Qinghua
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
  explicit Parser(Lexer *lexer);
  ~Parser() {
    ClearExprPool();
    ClearStmtPool();
    if (selfManagedLexer_) {
      delete lexer_;
      lexer_ = nullptr;
    }
  }

  // Parse statements.
  StmtPtr ParseCode();

  const std::string &filename() const { return lexer_->filename(); }

  void DumpAst();

private:
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
  StmtPtr ParseAugAssign();
  StmtPtr ParseReturn();
  Stmts ParserFunctionArgs();
  StmtPtr ParseFunctionDef();
  StmtPtr ParseClassDef();
  StmtPtr ParseStdCinCout();
  StmtPtr ParseIf();
  StmtPtr ParseFor();
  StmtPtr ParseWhile();
  StmtPtr ParseBlock();
  StmtPtr ParserCode();
  StmtPtr ParseModule();

  TokenConstPtr PreviousToken();
  TokenConstPtr NextToken();
  TokenConstPtr CurrentToken();
  TokenConstPtr GetToken();
  void RemoveToken();

  size_t TokenPos();
  void SetTokenPos(size_t pos);

  bool Finish();

  bool ParseStmts(StmtsPtr stmts);

  const std::string LineString(TokenConstPtr token);
  const std::string LineString();
  const std::string LineString(ExprConstPtr expr);
  const std::string LineString(StmtConstPtr stmt);

  Lexer *lexer_;
  bool selfManagedLexer_{false};

  StmtPtr module_{nullptr};
  size_t tokenPos_{0};
};
} // namespace parser

#endif // __PARSER_PARSER_H__