#ifndef __PARSER_COMPILER_COMPILER_H__
#define __PARSER_COMPILER_COMPILER_H__

#include <unordered_map>
#include <vector>

#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/parser.h"

namespace compiler {
using namespace parser;

#define INSTRUCTION(I) Inst_##I,
typedef enum OperatorId {
  Inst_Invalid,
#include "compiler/instruction.list"
  Inst_End,
} Inst;
#undef INSTRUCTION

struct InstCall {
  Inst inst;
  ssize_t offset;
  ssize_t lineno;
};
typedef InstCall *InstCallPtr;
typedef const InstCall *InstCallConstPtr;

#define TYPE(T) ConstType_##T,
enum ConstType {
  Invalid,
#include "lexer/literal_type.list"
  End,
};
#undef TYPE

struct Constant {
  ConstType type;
  std::string value;
};
typedef Constant *ConstantPtr;

class Compiler;
class CompilerNodeVisitor;
using StmtHandlerFunction = bool (Compiler::*)(StmtConstPtr);
using ExprHandlerFunction = bool (Compiler::*)(ExprConstPtr);

const char *GetInstStr(Inst inst);

class Compiler {
public:
  explicit Compiler(const std::string &filename);
  ~Compiler();

  void Compile();

  // Return true if stmt was handled, otherwise return false.
  bool CallStmtHandler(StmtConstPtr stmt) {
    return (this->*stmtHandlers_[stmt->type])(stmt);
  }

  // Return true if expr was handled, otherwise return false.
  bool CallExprHandler(ExprConstPtr expr) {
    return (this->*exprHandlers_[expr->type])(expr);
  }

  std::vector<InstCall> instructions() const { return instructions_; }
  void AddInstruction(const InstCall &inst) {
    instructions_.emplace_back(inst);
    lastLineno_ = inst.lineno;
  }

  void Dump();

private:
  void InitCompileHandlers();

  // Compile statement.
  bool CompileModule(StmtConstPtr stmt);
  bool CompileExpr(StmtConstPtr stmt);
  bool CompileAssign(StmtConstPtr stmt);
  bool CompileAugAssign(StmtConstPtr stmt);
  bool CompileReturn(StmtConstPtr stmt);
  bool CompileFunction(StmtConstPtr stmt);
  bool CompileClass(StmtConstPtr stmt);
  bool CompileBlock(StmtConstPtr stmt);
  bool CompileIf(StmtConstPtr stmt);
  bool CompileWhile(StmtConstPtr stmt);
  bool CompileFor(StmtConstPtr stmt);
  bool CompileBreak(StmtConstPtr stmt);
  bool CompileContinue(StmtConstPtr stmt);
  bool CompilePass(StmtConstPtr stmt);
  bool CompileImport(StmtConstPtr stmt);

  // Compile expression.
  bool CompileBinary(ExprConstPtr expr);
  bool CompileUnary(ExprConstPtr expr);
  bool CompileAttribute(ExprConstPtr expr);
  bool CompileSubscript(ExprConstPtr expr);
  bool CompileList(ExprConstPtr expr);
  bool CompileCall(ExprConstPtr expr);
  bool CompileName(ExprConstPtr expr);
  bool CompileLiteral(ExprConstPtr expr);

private:
  ssize_t FindVariableNameIndex(const std::string &name);

  Parser parser_;

  std::vector<std::string> variablePool_;
  std::vector<Constant> constantPool_;

  std::vector<InstCall> instructions_;
  ssize_t lastLineno_;
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
    if (!compiler_->CallStmtHandler(stmt)) {
      NodeVisitor::Visit(stmt);
    }
  }

  virtual void Visit(ExprConstPtr expr) override {
    if (expr == nullptr) {
    } else if (expr->type == ExprType_End) {
    } else if (expr->type == ExprType_Invalid) {
    }
    if (!compiler_->CallExprHandler(expr)) {
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

} // namespace compiler

#endif // __PARSER_COMPILER_COMPILER_H__