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
typedef enum InstType {
  Inst_Invalid,
#include "compiler/instruction.list"
  Inst_End,
} Inst;
#undef INSTRUCTION

#define TYPE(T) ConstType_##T,
enum ConstType {
  Invalid,
#include "lexer/literal_type.list"
  End,
};
#undef TYPE

class Compiler;
class CompilerNodeVisitor;
using StmtHandlerFunction = bool (Compiler::*)(StmtConstPtr);
using ExprHandlerFunction = bool (Compiler::*)(ExprConstPtr);
using StmtHandlerFunctions = std::unordered_map<StmtType, StmtHandlerFunction>;
using ExprHandlerFunctions = std::unordered_map<ExprType, ExprHandlerFunction>;

struct InstCall {
  Inst inst;      // Instruction type.
  ssize_t offset; // Extra information. Such as contant pool index, jump offset,
                  // and so on.
  ssize_t lineno; // Instruction lineno in the code.
};
typedef InstCall *InstCallPtr;
typedef const InstCall *InstCallConstPtr;

struct Constant {
  ConstType type;    // Constant type.
  std::string value; // Constant value.
};
typedef Constant *ConstantPtr;

struct Function {
  std::string name;              // Function name.
  ssize_t offset;                // Offset of function first instruction.
  std::vector<std::string> args; // Parameter names.
  std::vector<std::string> defs; // Parameter default values.
};
typedef Function *FunctionPtr;

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

  const std::vector<std::string> &symbolPool() const { return symbolPool_; }
  const std::vector<Constant> &constantPool() const { return constantPool_; }
  const std::vector<Function> &functionPool() const { return functionPool_; }
  const std::vector<InstCall> &instructions() const { return instructions_; }

  void AddInstruction(const InstCall &inst) {
    instructions_.emplace_back(inst);
    lastLineno_ = inst.lineno;
  }

  const std::string &filename() const { return parser_.filename(); }

  void Dump();

  ssize_t FindSymbolIndex(const std::string &name);
  ssize_t FindConstantIndex(const std::string &str);
  ssize_t FindFunctionIndex(const std::string &name);

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
  Parser parser_;
  CompilerNodeVisitor *walker_;
  ssize_t lastLineno_;

  // Compile result records start.
  // Do serialization or deserialization of them for compilation reuse.
  std::vector<std::string> symbolPool_;
  std::vector<Constant> constantPool_;
  std::vector<Function> functionPool_;
  std::vector<InstCall> instructions_;
  // Compile result records end.

  StmtHandlerFunctions stmtHandlers_; // Notice: Do not change.
  ExprHandlerFunctions exprHandlers_; // Notice: Do not change.
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

const char *GetInstStr(Inst inst);
} // namespace compiler

#endif // __PARSER_COMPILER_COMPILER_H__