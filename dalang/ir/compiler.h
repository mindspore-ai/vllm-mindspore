#ifndef __PARSER_IR_COMPILER_H__
#define __PARSER_IR_COMPILER_H__

#include <unordered_map>
#include <vector>

#include "ir/namespace.h"
#include "ir/node.h"
#include "parser/ast_visitor.h"
#include "parser/parser.h"

namespace ir {
using namespace parser;

class Compiler;
class CompilerNodeVisitor;
using StmtHandlerFunction = NsPtr (Compiler::*)(NsConstPtr, StmtConstPtr);
using ExprHandlerFunction = NodePtr (Compiler::*)(NsConstPtr, ExprConstPtr);

class Compiler {
public:
  explicit Compiler(const std::string &filename);
  ~Compiler();

  void Compile();

  // Return true if stmt was handled, otherwise return false.
  NsPtr CallStmtHandler(NsConstPtr ns, StmtConstPtr stmt) {
    return (this->*stmtHandlers_[stmt->type])(ns, stmt);
  }

  // Return true if expr was handled, otherwise return false.
  NodePtr CallExprHandler(NsConstPtr ns, ExprConstPtr expr) {
    return (this->*exprHandlers_[expr->type])(ns, expr);
  }

  NsConstPtr ns() const { return ns_; }

private:
  void InitCompileHandlers();

  // Compile statement.
  NsPtr CompileModule(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileExpr(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileAssign(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileAugAssign(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileReturn(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileFunction(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileClass(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileBlock(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileIf(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileWhile(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileFor(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileBreak(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileContinue(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompilePass(NsConstPtr ns, StmtConstPtr stmt);
  NsPtr CompileImport(NsConstPtr ns, StmtConstPtr stmt);

  // Compile expression.
  NodePtr CompileBinary(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileUnary(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileAttribute(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileSubscript(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileList(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileCall(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileName(NsConstPtr ns, ExprConstPtr expr);
  NodePtr CompileLiteral(NsConstPtr ns, ExprConstPtr expr);

private:
  Parser parser_;
  NsPtr ns_;
  std::unordered_map<StmtType, StmtHandlerFunction> stmtHandlers_;
  std::unordered_map<ExprType, ExprHandlerFunction> exprHandlers_;

  CompilerNodeVisitor *walker_;
};

class CompilerNodeVisitor : public NodeVisitor {
public:
  CompilerNodeVisitor(Compiler *compiler) : compiler_{compiler} {}
  virtual void Visit(StmtConstPtr stmt) override {
    if (stmt == nullptr) {
    } else if (stmt->type == StmtType_End) {
    } else if (stmt->type == StmtType_Invalid) {
    }
    if (!compiler_->CallStmtHandler(compiler_->ns(), stmt)) {
      NodeVisitor::Visit(stmt);
    }
  }

  virtual void Visit(ExprConstPtr expr) override {
    if (expr == nullptr) {
    } else if (expr->type == ExprType_End) {
    } else if (expr->type == ExprType_Invalid) {
    }
    if (!compiler_->CallExprHandler(compiler_->ns(), expr)) {
      NodeVisitor::Visit(expr);
    }
  }

  virtual void VisitList(size_t len, StmtConstPtr *stmtPtr) override {
    NodeVisitor::VisitList(len, stmtPtr);
  }

  virtual void VisitList(size_t len, ExprConstPtr *exprPtr) override {
    NodeVisitor::VisitList(len, exprPtr);
  }

private:
  Compiler *compiler_;
};
} // namespace ir

#endif // __PARSER_IR_COMPILER_H__