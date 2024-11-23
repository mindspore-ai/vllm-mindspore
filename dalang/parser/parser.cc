#include <filesystem>
#include <iomanip>

#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/expr.h"
#include "parser/parser.h"
#include "parser/stmt.h"

namespace parser {
Parser::Parser(const std::string &filename)
    : lexer_{Lexer(filename)},
      filename_{std::filesystem::canonical(std::filesystem::path(filename))
                    .string()} {}
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
  while (true) {
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
  while (true) {
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
            exit(1);
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
        exit(1);
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
#ifdef DEBUG
  LOG_OUT << LineString()
          << ", not match nothing, token: " << ToString(CurrentToken())
          << LOG_ENDL;
#endif
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
  return stmt;
}

StmtPtr Parser::ParseReturn() {
  if (StmtPattern::ReturnPattern::Match(CurrentToken())) {
    RemoveToken(); // return
    ExprConstPtr value = ParseExpr();
    return MakeReturnStmt(value);
  }
  return nullptr;
}

StmtPtr Parser::ParserFunctionDef() {
  // function
  if (StmtPattern::FunctionPattern::Match(CurrentToken())) {
    RemoveToken(); // function
    ExprConstPtr id = ParseIdentifier();
    ExprConstPtr args = ParseExpr();
    // {
    if (!StmtPattern::FunctionPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid function definition, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(1);
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
      exit(1);
    }
    RemoveToken(); // }
    return MakeFunctionStmt(id, args, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParserClassDef() {
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
      exit(1);
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
      exit(1);
    }
    RemoveToken(); // }
    return MakeClassStmt(id, bases, stmts);
  }
  return nullptr;
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
      exit(1);
    }
    // {
    if (!StmtPattern::IfPattern::MatchBodyStart(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid if statement, expected '{': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(1);
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
      exit(1);
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
        exit(1);
      }
      RemoveToken();                    // {
      (void)ParseStmts(&elseBodyStmts); // Not check result.
      // }
      if (!StmtPattern::IfPattern::MatchBodyEnd(CurrentToken())) {
        std::stringstream ss;
        ss << "warning: invalid else statement, expected '}': ";
        ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
        CompileMessage(LineString(), ss.str());
        exit(1);
      }
      RemoveToken(); // }
    }
    return MakeIfStmt(condition, ifBodyStmts, elseBodyStmts);
  }
  return nullptr;
}

StmtPtr Parser::ParserBlock() {
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
  // Class definition.
  stmt = ParserClassDef();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // Function definition.
  stmt = ParserFunctionDef();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  // If statement.
  stmt = ParseIf();
  if (stmt != nullptr || Finish()) {
    return stmt;
  }
  return nullptr;
}

bool Parser::ParseStmts(StmtsPtr stmts) {
  while (!Finish()) {
    TokenConstPtr lastToken = CurrentToken();
    StmtPtr stmt = ParserBlock();
    if (stmt != nullptr) {
      stmts->emplace_back(stmt);
    } else if (lastToken == CurrentToken()) { // Infinite loop, break.
      return false;
    } else {
#ifdef DEBUG
      LOG_OUT << "notice: handle partial tokens. last token: "
              << LineString(lastToken) << ": " << ToString(lastToken)
              << ", current token: " << LineString(CurrentToken()) << ": "
              << ToString(CurrentToken()) << LOG_ENDL;
#endif
    }
  }
  return true;
}

StmtsPtr Parser::ParseCode() {
  if (!ParseStmts(&stmts_)) {
    std::stringstream ss;
    ss << "warning: can not handle token: ";
    ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
    CompileMessage(LineString(), ss.str());
    exit(1);
  }
  return &stmts_;
}

constexpr int indentLen = 2;
void Parser::DumpAst() {
  class DumpNodeVisitor : public NodeVisitor {
  public:
    void Visit(StmtsConstPtr stmts) override {
      if (stmts != nullptr) {
        ss_ << "*Code [" << LOG_ENDL;
      }

      NodeVisitor::Visit(stmts); // Call parent Visit here.

      if (stmts != nullptr) {
        ss_ << ']' << LOG_ENDL;
      }
    }

    void Visit(StmtConstPtr stmt) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "Stmt/"
          << (stmt != nullptr ? ToString(stmt) : "null") << '(' << LOG_ENDL;

      NodeVisitor::Visit(stmt); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ')' << LOG_ENDL;
      --step_;
    }

    void Visit(ExprConstPtr expr) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "Expr/"
          << (expr != nullptr ? ToString(expr) : "null");
      if (expr->type != ExprType_Name && expr->type != ExprType_Literal) {
        ss_ << '(' << LOG_ENDL;
      }

      NodeVisitor::Visit(expr); // Call parent Visit here.

      if (expr->type != ExprType_Name && expr->type != ExprType_Literal) {
        ss_.seekp(-1, ss_.cur); // Remove a '\n'
        ss_ << ')';
      }
      ss_ << LOG_ENDL;
      --step_;
    }

    virtual void VisitList(size_t len, StmtConstPtr *stmtPtr) override {
      ++step_;
      const size_t indent = (step_ - 1) * indentLen;
      constexpr char indent_char = ' ';
      ss_ << std::string(indent, indent_char) << "Body[" << LOG_ENDL;

      NodeVisitor::VisitList(len, stmtPtr); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ']' << LOG_ENDL;
      --step_;
    }

    virtual void VisitList(size_t len, ExprConstPtr *exprPtr) override {
      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << "[" << LOG_ENDL;

      NodeVisitor::VisitList(len, exprPtr); // Call parent Visit here.

      ss_.seekp(-1, ss_.cur); // Remove a '\n'
      ss_ << ']' << LOG_ENDL;
    }

    const std::string dump() { return ss_.str(); }

  private:
    size_t step_{0};
    std::stringstream ss_;
  };
  auto visitor = DumpNodeVisitor();
  visitor.Visit(&stmts_);
  LOG_OUT << visitor.dump();
}

const std::string Parser::LineString(TokenConstPtr token) {
  return filename_ + ':' + std::to_string(token->lineStart) + ':' +
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
  return filename_ + ':' + std::to_string(expr->lineStart) + ':' +
         std::to_string(expr->columnStart + 1);
}

const std::string Parser::LineString(StmtConstPtr stmt) {
  return filename_ + ':' + std::to_string(stmt->lineStart) + ':' +
         std::to_string(stmt->columnStart + 1);
}
} // namespace parser