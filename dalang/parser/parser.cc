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
ExprPtr Parser::ParseExpr() { return ParseLogical(); }

ExprPtr Parser::ParseLogical() {
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

ExprPtr Parser::ParseUnary() { return ParseAttribute(); }
ExprPtr Parser::ParseAttribute() { return ParseCall(); }

ExprPtr Parser::ParseCall() {
  size_t reservedPos = TokenPos();
  // To match call pattern.
  ExprPtr func = ParseIdentifier();
  ExprPtr group = ParseGroup();
  // Match group but not call.
  if (func == nullptr && group != nullptr) {
    return group;
  }
  // If continuous call. such as [func][list]...[list]
  while (func != nullptr && group != nullptr) {
    func = MakeCallExpr(func, group);
    group = ParseGroup();
    if (group == nullptr) {
      return func;
    }
  }
  // Not match call or list, revert the token position, clear list expr memory
  // and re-parse for primary.
  SetTokenPos(reservedPos);
  if (group != nullptr) {
    ClearExprListMemory(group);
  }
  return ParsePrimary();
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
    } else if (ExprPattern::PrimaryPattern::MatchComment(CurrentToken())) {
      std::stringstream ss;
      ss << "notice: ignored comment: " + CurrentToken()->name;
      CompileMessage(LineString(), ss.str());
      RemoveToken();
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
  return nullptr;
}

ExprPtr Parser::ParseLiteral() {
  if (ExprPattern::PrimaryPattern::MatchLiteral(CurrentToken())) {
    return MakeLiteralExpr(GetToken());
  }
  return nullptr;
}

StmtPtr Parser::ParseStmt() {
  StmtPtr return_stmt = ParseReturn();
  if (return_stmt != nullptr) {
    return return_stmt;
  }
  return ParseAssign();
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
    if (!StmtPattern::FunctionPattern::MatchStart(CurrentToken())) {
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
    if (!StmtPattern::FunctionPattern::MatchEnd(CurrentToken())) {
      std::stringstream ss;
      ss << "warning: invalid function definition, expected '}': ";
      ss << (Finish() ? ToString(PreviousToken()) : ToString(CurrentToken()));
      CompileMessage(LineString(), ss.str());
      exit(1);
    }
    RemoveToken(); // }
    return MakeFunctionStmt(id, args, nullptr, stmts);
  }
  return nullptr;
}

StmtPtr Parser::ParserBlock() {
  StmtPtr stmt = ParseStmt();
  if (stmt != nullptr) {
    return stmt;
  }
  if (Finish()) {
    return nullptr;
  }
  return ParserFunctionDef();
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

void Parser::DumpAst() {
  class DumpNodeVisitor : public NodeVisitor {
  public:
    void visit(StmtsConstPtr stmts) override {
      if (stmts != nullptr) {
        LOG_OUT << "*Code {" << LOG_ENDL;
      }
      NodeVisitor::visit(stmts); // Call parent visit here.
      if (stmts != nullptr) {
        LOG_OUT << "}" << LOG_ENDL;
      }
    }
    void visit(StmtConstPtr stmt) override {
      ++step_;
      if (stmt != nullptr) {
        const size_t indent = step_ * 4;
        constexpr char indent_char = ' ';
        LOG_OUT << std::string(indent, indent_char) << "|-Stmt/"
                << ToString(stmt) << LOG_ENDL;
      }
      NodeVisitor::visit(stmt); // Call parent visit here.
      --step_;
    }
    void visit(ExprConstPtr expr) override {
      ++step_;
      if (expr != nullptr) {
        const size_t indent = step_ * 4;
        constexpr char indent_char = ' ';
        LOG_OUT << std::string(indent, indent_char) << "|-Expr/"
                << ToString(expr) << LOG_ENDL;
      }
      NodeVisitor::visit(expr); // Call parent visit here.
      --step_;
    }

  private:
    size_t step_{0};
  };
  DumpNodeVisitor().visit(&stmts_);
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