#include <filesystem>
#include <iomanip>

#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/parser.h"

namespace parser {
ExprPtr Parser::ParseExpr() { return ParseLogical(); }

ExprPtr Parser::ParseLogical() {
  ExprPtr left = ParseComparison();
  while (ExprPattern::LogicalPattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseComparison();
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseComparison() {
  ExprPtr left = ParseAdditive();
  while (ExprPattern::ComparisonPattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseAdditive();
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseAdditive() {
  ExprPtr left = ParseMultiplicative();
  while (ExprPattern::AdditivePattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseMultiplicative();
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseMultiplicative() {
  ExprPtr left = ParseUnary();
  while (ExprPattern::MultiplicativePattern::Match(CurrentToken())) {
    TokenConstPtr op = GetToken();
    ExprPtr right = ParseUnary();
    left = MakeBinaryExpr(op, left, right);
  }
  return left;
}

ExprPtr Parser::ParseUnary() { return ParseAttribute(); }
ExprPtr Parser::ParseAttribute() { return ParseCall(); }
ExprPtr Parser::ParseCall() { return ParseGroup(); }
ExprPtr Parser::ParseGroup() { return ParsePrimary(); }
ExprPtr Parser::ParsePrimary() {
  while (!Finish() && ExprPattern::PrimaryPattern::Match(CurrentToken())) {
    if (ExprPattern::PrimaryPattern::MatchIdentifier(CurrentToken())) {
      return MakeNameExpr(GetToken());
    } else if (ExprPattern::PrimaryPattern::MatchLiteral(CurrentToken())) {
      return MakeLiteralExpr(GetToken());
    } else if (ExprPattern::PrimaryPattern::MatchKeyword(CurrentToken())) {
      CompileMessage(LineString(),
                     std::string("warning: not support keyword: ") +
                         ToStr(CurrentToken()->data.kw));
      GetToken(); // unused.
      exit(1);
    } else if (ExprPattern::PrimaryPattern::MatchComment(CurrentToken())) {
      CompileMessage(LineString(),
                     "notice: ignored comment: " + CurrentToken()->name);
      GetToken(); // unused.
    }
  }
  CompileMessage(LineString(),
                 std::string("warning: wrong DaLang code. token: ") +
                     ToStr(CurrentToken()));
  GetToken(); // unused.
  exit(1);
}

StmtPtr Parser::ParseStmt() {
  StmtPtr return_stmt = ParseReturn();
  if (return_stmt != nullptr)
    return return_stmt;
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
    TokenConstPtr assign = GetToken(); // unused.
    ExprConstPtr value = ParseExpr();
    return MakeAssignStmt(target, value);
  }
  return stmt;
}

StmtPtr Parser::ParseReturn() {
  if (StmtPattern::ReturnPattern::Match(CurrentToken())) {
    TokenConstPtr return_ = GetToken(); // unused.
    ExprConstPtr value = ParseExpr();
    return MakeReturnStmt(value);
  }
  return nullptr;
}

StmtsPtr Parser::ParseStmts() {
  while (!Finish()) {
    StmtPtr stmt = ParseStmt();
    if (stmt != nullptr) {
      stmts_.emplace_back(stmt);
    }
  }
  return &stmts_;
}

void Parser::DumpAst() {
  class DumpNodeVisitor : public NodeVisitor {
  public:
    void visit(StmtsConstPtr stmts) override {
      if (stmts != nullptr) {
        LOG_OUT << "*Stmts {" << LOG_ENDL;
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
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(token->lineStart) + ':' +
         std::to_string(token->columnStart);
}

const std::string Parser::LineString() {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(CurrentToken()->lineStart) + ':' +
         std::to_string(CurrentToken()->columnStart);
}

const std::string Parser::LineString(ExprConstPtr expr) {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(expr->lineStart) + ':' +
         std::to_string(expr->columnStart);
}

const std::string Parser::LineString(StmtConstPtr stmt) {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(stmt->lineStart) + ':' +
         std::to_string(stmt->columnStart);
}
} // namespace parser