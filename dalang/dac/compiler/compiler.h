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

#ifndef __PARSER_COMPILER_COMPILER_H__
#define __PARSER_COMPILER_COMPILER_H__

#include <stack>
#include <unordered_map>
#include <vector>

#include "parser/ast_node.h"
#include "parser/ast_visitor.h"
#include "parser/parser.h"

namespace compiler {
using namespace parser;

#define INSTRUCTION(I) Inst_##I,
typedef enum InstType {
#include "compiler/instruction.list"
  Inst_End,
} Inst;
#undef INSTRUCTION

#define TYPE(T) ConstType_##T,
enum ConstType {
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

enum CodeType { CodeBlock, CodeFunction, CodeGraph, CodeModule, CodeEnd };
const char *ToStr(CodeType type);

struct Code {
  CodeType type;                    // Type of block/function/graph/module.
  std::string name;                 // Function, graph or module name.
  std::vector<std::string> symbols; // Symbol pool in the namespace.
  std::vector<Constant> constants;  // Constant pool in the namespace.
  std::vector<InstCall> insts;      // Instructions in the namespace.
  std::vector<std::string> args;    // Parameter names.
  std::vector<std::string> defs;    // Parameter default values.
};
typedef Code *CodePtr;

class Compiler {
public:
  explicit Compiler(const std::string &filename);
  explicit Compiler(Parser *parser);
  ~Compiler();

  void Compile();

  const std::string &filename() const { return parser_->filename(); }
  const std::vector<Code> &codes() const { return codes_; }

  // Return true if stmt was handled, otherwise return false.
  bool CallStmtHandler(StmtConstPtr stmt) {
    return (this->*stmtHandlers_[stmt->type])(stmt);
  }

  // Return true if expr was handled, otherwise return false.
  bool CallExprHandler(ExprConstPtr expr) {
    return (this->*exprHandlers_[expr->type])(expr);
  }

  void Dump();

private:
  void Init();
  void InitCompileHandlers();

  // Compile statement.
  bool CompileModule(StmtConstPtr stmt);
  bool CompileExpr(StmtConstPtr stmt);
  bool CompileAssign(StmtConstPtr stmt);
  bool CompileAugAssign(StmtConstPtr stmt);
  bool CompileReturn(StmtConstPtr stmt);
  bool CompileGraph(StmtConstPtr stmt);
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
  bool CompileStdCin(StmtConstPtr stmt);
  bool CompileStdCout(StmtConstPtr stmt);

  // Compile expression.
  bool CompileBinary(ExprConstPtr expr);
  bool CompileUnary(ExprConstPtr expr);
  bool CompileAttribute(ExprConstPtr expr);
  bool CompileSubscript(ExprConstPtr expr);
  bool CompileList(ExprConstPtr expr);
  bool CompileCall(ExprConstPtr expr);
  bool CompileName(ExprConstPtr expr);
  bool CompileLiteral(ExprConstPtr expr);

  size_t CurrentCodeIndex() { return codeStack_.top(); }
  Code &CurrentCode() { return codes_[codeStack_.top()]; }
  Code &code(size_t index) { return codes_[index]; }

  std::vector<std::string> &symbolPool(size_t index) {
    return code(index).symbols;
  }
  std::vector<Constant> &constantPool(size_t index) {
    return code(index).constants;
  }
  std::vector<InstCall> &instructions(size_t index) {
    return code(index).insts;
  }
  void AddInstruction(const InstCall &inst) {
    CurrentCode().insts.emplace_back(inst);
    lastInst_ = inst;
  }

  ssize_t FindSymbolIndex(const std::string &name);
  ssize_t FindGlobalSymbolIndex(const std::string &name);
  ssize_t FindConstantIndex(const std::string &str);

private:
  Parser *parser_;
  bool selfManagedParser_{false};
  CompilerNodeVisitor *walker_;
  InstCall lastInst_;
  std::stack<size_t> codeStack_;

  ssize_t intrinsicSize_{0};

  // Compile result records start.
  // Do serialization or deserialization of them for compilation reuse.
  std::vector<Code> codes_;
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
    }
    if (!compiler_->CallStmtHandler(stmt)) {
      NodeVisitor::Visit(stmt);
    }
  }

  virtual void Visit(ExprConstPtr expr) override {
    if (expr == nullptr) {
    } else if (expr->type == ExprType_End) {
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