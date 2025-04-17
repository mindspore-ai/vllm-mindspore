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

#include "ir/compiler.h"

#include "common/common.h"

namespace ir {
Compiler::Compiler(const std::string &filename)
    : parser_{Parser(filename)}, walker_{new CompilerNodeVisitor(this)} {
  InitCompileHandlers();
}

Compiler::~Compiler() {
  ClearNamespacePool();
  ClearBlockPool();
  ClearFuncPool();
  delete walker_;
}

void Compiler::Compile() {
  StmtPtr module = parser_.ParseCode();
  ns_ = NewNamespace();
  if (walker_ == nullptr) {
    LOG_ERROR << "AST walker should not be null.";
    exit(EXIT_FAILURE);
  }
  walker_->Visit(module);
}

NsPtr Compiler::CompileModule(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileExpr(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt) << "/" << ToString(stmt->stmt.Expr.value);
  return nullptr;
}

NsPtr Compiler::CompileAssign(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileAugAssign(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileReturn(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileGraph(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileFunction(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  FuncPtr func = NewFunc();
  CHECK_IF_NULL(stmt);
  const auto &funcStmt = stmt->stmt.Function;
  // Function name.
  ExprConstPtr expr = funcStmt.name;
  CHECK_IF_NULL(expr);
  CHECK_IF_NULL(expr->expr.Name.identifier);
  const auto funcName = *(expr->expr.Name.identifier);
  func->setName(funcName);
  LOG_OUT << "func name: " << funcName;
  // Function parameters.
  LOG_OUT << "func args len: " << funcStmt.argsLen;
  for (size_t i = 0; i < funcStmt.argsLen; ++i) {
    const auto &argStmt = funcStmt.args[i];
    LOG_OUT << "func args[" << i << "]: " << ToString(argStmt);
    if (argStmt->type == StmtType_Expr &&
        argStmt->stmt.Expr.value->type == ExprType_Name) {
      const auto &argName = *argStmt->stmt.Expr.value->expr.Name.identifier;
      func->AddParameter(argName);
    } else if (argStmt->type == StmtType_Assign &&
               argStmt->stmt.Assign.target->type == ExprType_Name &&
               argStmt->stmt.Assign.value->type == ExprType_Literal) {
      const auto &argName = *argStmt->stmt.Assign.target->expr.Name.identifier;
      const auto &literal = argStmt->stmt.Assign.value->expr.Literal;
      const auto &defaultParam = *literal.value;
      func->AddParameter(argName, defaultParam);
    }
  }
  // Function body.
  LOG_OUT << "func body len: " << funcStmt.len;
  for (size_t i = 0; i < funcStmt.len; ++i) {
    const auto &stmt = funcStmt.body[i];
    NsConstPtr subNs = CallStmtHandler(ns, stmt);
    // func->AddNs(subNs);
    LOG_OUT << "func body[" << i << "]: " << ToString(stmt)
            << ", ns: " << subNs;
  }
  LOG_OUT << "func: " << func->ToString();
  return func;
}

NsPtr Compiler::CompileClass(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileBlock(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileIf(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileWhile(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileFor(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileBreak(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileContinue(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompilePass(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileImport(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileStdCin(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NsPtr Compiler::CompileStdCout(NsConstPtr ns, StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return nullptr;
}

NodePtr Compiler::CompileBinary(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileUnary(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileAttribute(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileSubscript(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileList(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileCall(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileName(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

NodePtr Compiler::CompileLiteral(NsConstPtr ns, ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return nullptr;
}

#define STMT(type) stmtHandlers_[StmtType_##type] = &Compiler::Compile##type;
#define EXPR(type) exprHandlers_[ExprType_##type] = &Compiler::Compile##type;
void Compiler::InitCompileHandlers() {
#include "parser/expr.list"
#include "parser/stmt.list"
}
#undef STMT
#undef EXPR
} // namespace ir