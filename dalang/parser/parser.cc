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

#include <filesystem>
#include <iomanip>

#include "common/common.h"
#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/expr.h"
#include "parser/parser.h"
#include "parser/stmt.h"

#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT

namespace parser {
Parser::Parser(const std::string &filename) : selfManagedLexer_{true} {
  lexer_ = new Lexer(filename);
}

Parser::Parser(Lexer *lexer) : lexer_{lexer}, selfManagedLexer_{false} {}

ExprPtr Parser::ParseExpr() {
  // Ignore all comments.
  while (ExprPattern::PrimaryPattern::MatchComment(CurrentToken())) {
#ifdef DEBUG
    std::stringstream ss;
    ss << "notice: ignored comment: " + CurrentToken()->name;
    CompileMessage(LineString(), ss.str());
#endif
    RemoveToken();
  }
  return ParseLogical();
}

ExprPtr Parser::ParseLogicalOr() {
  ExprPtr left = ParseComparison();
  if (left == nullptr) {
    return nullptr;
  }
  while (ExprPattern::LogicalPattern::MatchOr(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseLogicalAnd();
    if (right == nullptr) {
      return nullptr;
    }
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseLogicalAnd() {
  ExprPtr left = ParseComparison();
  if (left == nullptr) {
    return nullptr;
  }
  while (ExprPattern::LogicalPattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseComparison();
    if (right == nullptr) {
      return nullptr;
    }
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseLogical() { return ParseLogicalOr(); }

ExprPtr Parser::ParseComparison() {
  ExprPtr left = ParseAdditive();
  if (left == nullptr) {
    return nullptr;
  }
  while (ExprPattern::ComparisonPattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseAdditive();
    if (right == nullptr) {
      return nullptr;
    }
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseAdditive() {
  ExprPtr left = ParseMultiplicative();
  if (left == nullptr) {
    return nullptr;
  }
  while (ExprPattern::AdditivePattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseMultiplicative();
    if (right == nullptr) {
      return nullptr;
    }
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseMultiplicative() {
  ExprPtr left = ParseUnary();
  if (left == nullptr) {
    return nullptr;
  }
  while (ExprPattern::MultiplicativePattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseUnary();
    if (right == nullptr) {
      return nullptr;
    }
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseUnary() { return ParseCallAndAttribute(); }

ExprPtr Parser::ParseCallAndAttribute() {
  ExprPtr expr = ParseIdentifier();
  if (expr == nullptr) {
    expr = ParseGroup();
    if (expr != nullptr) {
      return expr;
    }
    expr = ParsePrimary();
    return expr;
  }
  // Start parse call and attribute combination.
  ExprPtr call = expr;
  for (EVER) {
    call = ParseCall(call);
    ExprPtr attr = ParseAttribute(call);
    if (attr == call) {
      return call;
    }
    call = attr;
  }
}

// Return input as output if not match any token.
ExprPtr Parser::ParseAttribute(ExprPtr entity) {
  if (entity == nullptr) {
    return nullptr;
  }
  ExprPtr attr = entity;
  while (ExprPattern::AttributePattern::Match(CurrentToken())) {
    RemoveToken(); // .
    ExprPtr id = ParseIdentifier();
    attr = MakeAttributeExpr(attr, id);
  }
  return attr;
}

// Return input as output if not match any token.
ExprPtr Parser::ParseCall(ExprPtr func) {
  if (func == nullptr) {
    return nullptr;
  }
  // If continuous call. such as [func][list]...[list]
  for (EVER) {
    ExprPtr group = ParseGroup();
    if (group == nullptr) {
      return func;
    }
    func = MakeCallExpr(func, group);
  }
}

ExprPtr Parser::ParseGroup() {
  if (ExprPattern::GroupPattern::Match(CurrentToken())) {
    Exprs elements;
    if (ExprPattern::GroupPattern::MatchStart(CurrentToken())) {
      TokenConstPtr start = GetToken(); // (
      ExprPtr expr = ParseExpr();
      if (expr != nullptr) { // Not empty list.
        // The first element.
        (void)elements.emplace_back(expr);

        while (ExprPattern::GroupPattern::MatchSplit(CurrentToken())) {
          RemoveToken(); // ,
          // The middle and last elements.
          expr = ParseExpr();
          if (expr != nullptr) {
            (void)elements.emplace_back(expr);
          } else { // Abnormal group expression.
            std::stringstream ss;
            ss << "warning: invalid list. unexcepted token: ";
            ss << (Finish() ? ToString(PreviousToken())
                            : ToString(CurrentToken()));
            CompileMessage(LineString(), ss.str());
            exit(EXIT_FAILURE);
          }
        }
      }
      if (ExprPattern::GroupPattern::MatchEnd(CurrentToken())) {
        TokenConstPtr end = GetToken(); // )
        return MakeListExpr(start, end, elements);
      } else { // Abnormal group expression.
        std::stringstream ss;
        ss << "warning: invalid list ending. unrecognized token: ";
        ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
        CompileMessage(LineString(), ss.str());
        exit(EXIT_FAILURE);
      }
    }
  }
  return nullptr;
}

ExprPtr Parser::ParsePrimary() {
  while (!Finish() && ExprPattern::PrimaryPattern::Match(CurrentToken())) {
    ExprPtr expr = ParseIdentifier();
    if (expr != nullptr) {
      return expr;
    }
    expr = ParseLiteral();
    if (expr != nullptr) {
      return expr;
    }
    if (ExprPattern::PrimaryPattern::MatchKeyword(CurrentToken())) {
      return nullptr;
    }
  }
  LOG_OUT << LineString()
          << ", not match nothing, token: " << ToString(CurrentToken());
  return nullptr;
}

ExprPtr Parser::ParseIdentifier() {
  if (ExprPattern::PrimaryPattern::MatchIdentifier(CurrentToken())) {
    return MakeNameExpr(GetToken());
  }
  if (ExprPattern::PrimaryPattern::MatchKeywordThis(CurrentToken())) {
    return MakeNameExpr(GetToken());
  }
  return nullptr;
}

ExprPtr Parser::ParseLiteral() {
  if (ExprPattern::PrimaryPattern::MatchLiteral(CurrentToken())) {
    return MakeLiteralExpr(GetToken());
  }
  return nullptr;
}

StmtPtr Parser::ParseStmtExpr() {
  ExprPtr value = ParseExpr();
  if (value == nullptr) {
    return nullptr;
  }
  return MakeExprStmt(value);
}

StmtPtr Parser::ParseAssign() {
  size_t reservedTokenPos = TokenPos();
  StmtPtr stmt = ParseStmtExpr();
  if (stmt == nullptr) {
    return nullptr;
  }
  if (StmtPattern::AssignPattern::Match(CurrentToken())) {
    ExprConstPtr target = stmt->stmt.Expr.value;
    RemoveToken(); // =
    ExprConstPtr value = ParseExpr();
    if (value == nullptr) {
      return nullptr;
    }
    return MakeAssignStmt(target, value);
  }
  SetTokenPos(reservedTokenPos);
  return nullptr;
}

// Augmented Assign, such as +=, -=, *=, /=, %=
StmtPtr Parser::ParseAugAssign() {
  size_t reservedTokenPos = TokenPos();
  // Left target.
  StmtPtr stmt = ParseStmtExpr();
  if (stmt == nullptr) {
    return nullptr;
  }
  if (StmtPattern::AugAssignPattern::Match(CurrentToken())) {
    OpId op = GetToken()->data.op; // .=
    ExprConstPtr value = ParseExpr();
    if (value == nullptr) {
      return nullptr;
    }
    ExprConstPtr target = stmt->stmt.Expr.value;
    return MakeAugAssignStmt(target, op, value);
  }
  SetTokenPos(reservedTokenPos);
  return nullptr;
}

StmtPtr Parser::ParseReturn() {
  if (StmtPattern::ReturnPattern::Match(CurrentToken())) {
    RemoveToken(); // return
    ExprConstPtr value = ParseExpr();
    return MakeReturnStmt(value);
  }
  return nullptr;
}

Stmts Parser::ParserFunctionArgs() {
  Stmts args;
  // (
  if (StmtPattern::FunctionPattern::MatchArgsStart(CurrentToken())) {
    RemoveToken(); // (
    for (EVER) {
      StmtPtr arg = ParseAssign();
      if (arg == nullptr) {
        arg = ParseStmtExpr();
      }
      if (arg != nullptr) {
        (void)args.emplace_back(arg);
      }
      // )
      if (StmtPattern::FunctionPattern::MatchArgsEnd(CurrentToken())) {
        RemoveToken(); // )
        break;
      }
      // ,
      if (!StmtPattern::FunctionPattern::MatchArgsSeparator(CurrentToken())) {
        std::stringstream ss;
        ss << "warning: invalid function arguments, expected ',' or ')': ";
        ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
        CompileMessage(LineString(), ss.str());
        exit(EXIT_FAILURE);
      }
      RemoveToken(); // ,
    }
  }
  return args;
}

StmtPtr Parser::ParseFunctionDef() {
  // function
  if (StmtPattern::FunctionPattern::Match(CurrentToken())) {
    RemoveToken(); // function
    ExprConstPtr id = ParseIdentifier();
    Stmts args = ParserFunctionArgs();
    // {
    if (!StmtPattern::FunctionPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid function definition, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // {
    Stmts stmts;
    (void)ParseStmts(&stmts); // Not check result.
    // }
    if (!StmtPattern::FunctionPattern::MatchBodyEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid function definition, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // }
    return MakeFunctionStmt(id, args, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParseClassDef() {
  // class
  if (StmtPattern::ClassPattern::Match(CurrentToken())) {
    RemoveToken(); // class
    ExprConstPtr id = ParseIdentifier();
    ExprConstPtr bases = ParseExpr();
    // {
    if (!StmtPattern::ClassPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid class definition, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // {
    Stmts stmts;
    (void)ParseStmts(&stmts); // Not check result.
    // }
    if (!StmtPattern::ClassPattern::MatchBodyEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid class definition, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // }
    return MakeClassStmt(id, bases, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParseBlock() {
  // block
  // {
  if (!StmtPattern::BlockPattern::MatchBodyStart(CurrentToken())) {
    return nullptr;
  }
  RemoveToken(); // {
  Stmts stmts;
  (void)ParseStmts(&stmts); // Not check result.
  // }
  if (!StmtPattern::BlockPattern::MatchBodyEnd(CurrentToken())) {
    std::stringstream ss;
    ss << "warning: invalid code block, expected '}': ";
    ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
    CompileMessage(LineString(), ss.str());
    exit(EXIT_FAILURE);
  }
  RemoveToken(); // }
  return MakeBlockStmt(stmts);
}

StmtPtr Parser::ParseIf() {
  // if
  if (StmtPattern::IfPattern::MatchIf(CurrentToken())) {
    RemoveToken();                        // if
    ExprConstPtr condition = ParseExpr(); // condition
    if (condition == nullptr) {
      std::stringstream ss;
      ss << "warning: invalid if statement, expected a condition expression: ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    // {
    if (!StmtPattern::IfPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid if statement, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // {
    Stmts ifBodyStmts;
    (void)ParseStmts(&ifBodyStmts); // Not check result.
    // }
    if (!StmtPattern::IfPattern::MatchBodyEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid if statement, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // }

    // else
    Stmts elseBodyStmts;
    if (StmtPattern::IfPattern::MatchElse(CurrentToken())) {
      RemoveToken(); // else

      // else if
      if (StmtPattern::IfPattern::MatchIf(CurrentToken())) {
        auto elseIfStmt = ParseIf();
        (void)elseBodyStmts.emplace_back(elseIfStmt);
        return MakeIfStmt(condition, ifBodyStmts, elseBodyStmts);
      }

      // {
      if (!StmtPattern::IfPattern::MatchBodyStart(CurrentToken())) {
        std::stringstream ss;
        ss << "warning: invalid else statement, expected '{': ";
        ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
        CompileMessage(LineString(), ss.str());
        exit(EXIT_FAILURE);
      }
      RemoveToken();                    // {
      (void)ParseStmts(&elseBodyStmts); // Not check result.
      // }
      if (!StmtPattern::IfPattern::MatchBodyEnd(CurrentToken())) {
        std::stringstream ss;
        ss << "warning: invalid else statement, expected '}': ";
        ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
        CompileMessage(LineString(), ss.str());
        exit(EXIT_FAILURE);
      }
      RemoveToken(); // }
    }
    return MakeIfStmt(condition, ifBodyStmts, elseBodyStmts);
  }
  return nullptr;
}

StmtPtr Parser::ParseFor() {
  // for
  if (StmtPattern::ForPattern::Match(CurrentToken())) {
    RemoveToken();                      // for
    ExprConstPtr element = ParseExpr(); // element
    if (element == nullptr) {
      std::stringstream ss;
      ss << "warning: invalid for statement, expected an element expression: ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    // :
    if (!StmtPattern::ForPattern::MatchIteratorSeparator(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid for statement, expected ':': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken();                       // :
    ExprConstPtr iterator = ParseExpr(); // iterator
    if (iterator == nullptr) {
      std::stringstream ss;
      ss << "warning: invalid for statement, expected an iterator expression: ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    // {
    if (!StmtPattern::ForPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid for statement, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // {
    Stmts stmts;
    (void)ParseStmts(&stmts); // Not check result.
    // }
    if (!StmtPattern::ForPattern::MatchBodyEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid for statement, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // }
    return MakeForStmt(element, iterator, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParseWhile() {
  // while
  if (StmtPattern::WhilePattern::Match(CurrentToken())) {
    RemoveToken();                   // while
    ExprConstPtr cond = ParseExpr(); // condition
    if (cond == nullptr) {
      std::stringstream ss;
      ss << "warning: invalid while statement, expected a condition "
            "expression: ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    // {
    if (!StmtPattern::WhilePattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid while statement, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // {
    Stmts stmts;
    (void)ParseStmts(&stmts); // Not check result.
    // }
    if (!StmtPattern::WhilePattern::MatchBodyEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid while statement, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(EXIT_FAILURE);
    }
    RemoveToken(); // }
    return MakeWhileStmt(cond, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParseStdCinCout() {
  // >: or <:
  if (StmtPattern::StdCinCoutPattern::Match(CurrentToken())) {
    const bool isStdCin =
        StmtPattern::StdCinCoutPattern::MatchStdCin(CurrentToken());
    RemoveToken(); // >: or <:
    ExprConstPtr value = ParseExpr();
    if (isStdCin) {
      return MakeStdCinStmt(value);
    } else {
      return MakeStdCoutStmt(value);
    }
  }
  return nullptr;
}

StmtPtr Parser::ParserCode() {
  // Return statement.
  StmtPtr stmt = ParseReturn();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Assign statement.
  stmt = ParseAssign();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Augmented Assign statement.
  stmt = ParseAugAssign();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Class definition.
  stmt = ParseClassDef();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Function definition.
  stmt = ParseFunctionDef();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // If statement.
  stmt = ParseIf();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // For statement.
  stmt = ParseFor();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // While statement.
  stmt = ParseWhile();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Block statement.
  stmt = ParseBlock();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Block std cin&cout.
  stmt = ParseStdCinCout();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  return ParseStmtExpr();
}

bool Parser::ParseStmts(StmtsPtr stmts) {
  while (!Finish()) {
    TokenConstPtr lastToken = CurrentToken();
    StmtPtr stmt = ParserCode();
    if (stmt != nullptr) {
      LOG_OUT << "stmt: " << ToString(stmt);
      (void)stmts->emplace_back(std::move(stmt));
    } else if (lastToken == CurrentToken()) { // Infinite loop, break.
      return false;
    } else {
      if (CurrentToken() != nullptr) {
        LOG_OUT << "notice: handle partial tokens. last token: "
                << LineString(lastToken) << ": " << ToString(lastToken)
                << ", current token: " << LineString(CurrentToken()) << ": "
                << ToString(CurrentToken());
      }
    }
  }
  return true;
}

StmtPtr Parser::ParseModule() {
  Stmts stmts;
  if (!ParseStmts(&stmts)) {
    std::stringstream ss;
    ss << "warning: can not handle token: ";
    ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
    CompileMessage(LineString(), ss.str());
    exit(EXIT_FAILURE);
  }
  return MakeModuleStmt(stmts);
}

StmtPtr Parser::ParseCode() {
  if (module_ == nullptr) {
    module_ = ParseModule();
  }
  return module_;
}

TokenConstPtr Parser::PreviousToken() {
  if (tokenPos_ - 1 >= lexer_->Tokens().size()) {
    return nullptr;
  }
  return &lexer_->Tokens()[tokenPos_ - 1];
}

TokenConstPtr Parser::NextToken() {
  if (tokenPos_ + 1 >= lexer_->Tokens().size()) {
    return nullptr;
  }
  return &lexer_->Tokens()[tokenPos_ + 1];
}

TokenConstPtr Parser::CurrentToken() {
  if (tokenPos_ >= lexer_->Tokens().size()) {
    return nullptr;
  }
  return &lexer_->Tokens()[tokenPos_];
}

TokenConstPtr Parser::GetToken() {
  if (tokenPos_ >= lexer_->Tokens().size()) {
    CompileMessage(LineString(&lexer_->Tokens().back()),
                   "warning: tokens were exhaused");
    exit(EXIT_FAILURE);
  }
  const auto *token = &lexer_->Tokens()[tokenPos_];
  ++tokenPos_;
  return token;
}

void Parser::RemoveToken() {
  if (tokenPos_ >= lexer_->Tokens().size()) {
    CompileMessage(LineString(&lexer_->Tokens().back()),
                   "warning: tokens were exhaused");
    exit(EXIT_FAILURE);
  }
  ++tokenPos_;
}

size_t Parser::TokenPos() { return tokenPos_; }

void Parser::SetTokenPos(size_t pos) { tokenPos_ = pos; }

bool Parser::Finish() { return tokenPos_ == lexer_->Tokens().size(); }

const std::string Parser::LineString(TokenConstPtr token) {
  return filename() + ':' + std::to_string(token->lineStart) + ':' +
         std::to_string(token->columnStart + 1);
}

const std::string Parser::LineString() {
  TokenConstPtr token;
  if (Finish()) {
    token = PreviousToken();
  } else {
    token = CurrentToken();
  }
  return LineString(token);
}

const std::string Parser::LineString(ExprConstPtr expr) {
  return filename() + ':' + std::to_string(expr->lineStart) + ':' +
         std::to_string(expr->columnStart + 1);
}

const std::string Parser::LineString(StmtConstPtr stmt) {
  return filename() + ':' + std::to_string(stmt->lineStart) + ':' +
         std::to_string(stmt->columnStart + 1);
}

constexpr int indentLen = 2;
void Parser::DumpAst() {
  class DumpNodeVisitor : public NodeVisitor {
  public:
    void Visit(StmtConstPtr stmt) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "$"
          << (stmt != nullptr ? ToString(stmt) : "null") << '(' << ENDL;

      NodeVisitor::Visit(stmt); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ')' << ENDL;
      --step_;
    }

    void Visit(ExprConstPtr expr) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "!"
          << (expr != nullptr ? ToString(expr) : "null");
      if (expr != nullptr && expr->type != ExprType_Name &&
          expr->type != ExprType_Literal) {
        ss_ << '(' << ENDL;
      }

      NodeVisitor::Visit(expr); // Call parent Visit here.

      if (expr != nullptr && expr->type != ExprType_Name &&
          expr->type != ExprType_Literal) {
        ss_.seekp(-1, ss_.cur); // Remove a '\n'
        ss_ << ')';
      }
      ss_ << ENDL;
      --step_;
    }

    virtual void VisitList(size_t len, StmtConstPtr *stmtPtr) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "Body[" << ENDL;

      NodeVisitor::VisitList(len, stmtPtr); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ']' << ENDL;
      --step_;
    }

    virtual void VisitList(size_t len, ExprConstPtr *exprPtr) override {
      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << "[" << ENDL;

      NodeVisitor::VisitList(len, exprPtr); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ']' << ENDL;
    }

    const std::string dump() { return ss_.str(); }

  private:
    size_t step_{0};
    std::stringstream ss_;
  };
  auto visitor = DumpNodeVisitor();
  visitor.Visit(module_);
  std::cout << "--------------------" << std::endl;
  std::cout << "------- AST --------" << std::endl;
  std::cout << visitor.dump();
}
} // namespace parser