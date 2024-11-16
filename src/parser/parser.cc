#include <filesystem>
#include <iomanip>

#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/parser.h"

namespace parser {
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
      TokenConstPtr start = GetToken();
      ExprPtr expr = ParseExpr();
      if (expr != nullptr) { // Not empty list.
        // The first element.
        (void)elements.emplace_back(expr);

        while (ExprPattern::GroupPattern::MatchSplit(CurrentToken())) {
          TokenConstPtr split = GetToken(); // unused.
          // The middle elements.
          expr = ParseExpr();
          (void)elements.emplace_back(expr);
        }

        // The last element.
        expr = ParseExpr();
        if (expr != nullptr) {
          (void)elements.emplace_back(expr);
        }
      }
      if (ExprPattern::GroupPattern::MatchEnd(CurrentToken())) {
        TokenConstPtr end = GetToken();
        return MakeListExpr(start, end, elements);
      } else { // Abnormal group expression.
        CompileMessage(LineString(), std::string("warning: invalid list: ") +
                                         ToStr(CurrentToken()));
        GetToken(); // unused.
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
  // CompileMessage(LineString(),
  //                std::string("notice: wrong DaLang code. token: ") +
  //                    ToString(CurrentToken()));
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
    TokenConstPtr assign = GetToken(); // unused.
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
    TokenConstPtr return_ = GetToken(); // unused.
    ExprConstPtr value = ParseExpr();
    return MakeReturnStmt(value);
  }
  return nullptr;
}

StmtsPtr Parser::ParseStmts() {
  while (!Finish()) {
    TokenConstPtr lastToken = CurrentToken();
    StmtPtr stmt = ParseStmt();
    if (stmt != nullptr) {
      stmts_.emplace_back(stmt);
    } else if (lastToken == CurrentToken()) { // Infinite loop, exit.
      CompileMessage(LineString(),
                     std::string("warning: can not handle token: ") +
                         ToString(CurrentToken()));
      exit(1);
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
         std::to_string(token->columnStart + 1);
}

const std::string Parser::LineString() {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(CurrentToken()->lineStart) + ':' +
         std::to_string(CurrentToken()->columnStart + 1);
}

const std::string Parser::LineString(ExprConstPtr expr) {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(expr->lineStart) + ':' +
         std::to_string(expr->columnStart + 1);
}

const std::string Parser::LineString(StmtConstPtr stmt) {
  return std::filesystem::canonical(std::filesystem::path(filename_)).string() +
         ':' + std::to_string(stmt->lineStart) + ':' +
         std::to_string(stmt->columnStart + 1);
}
} // namespace parser