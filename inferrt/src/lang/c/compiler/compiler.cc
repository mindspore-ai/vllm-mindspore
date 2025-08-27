/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "lang/c/compiler/compiler.h"

#include "common/common.h"
#include "ops/op_def/ops_name.h"

using namespace mrt;

#define TO_STR(s) #s
#define INSTRUCTION(I) TO_STR(I),
const char *_insts[]{// Inst strings.
#include "lang/c/compiler/instruction.list"
                     "End"};
#undef INSTRUCTION

namespace da {
namespace compiler {
const char *ToStr(CodeType type) {
  switch (type) {
    case CodeBlock: {
      return "block";
    }
    case CodeFunction: {
      return "function";
    }
    case CodeGraph: {
      return "graph";
    }
    case CodeModule: {
      return "module";
    }
    default:
      return "<unknown>";
  }
}

const char *GetInstStr(Inst inst) {
  if (inst >= Inst_End) {
    LOG_ERROR << "inst is abnormal, " << inst << ", " << Inst_End;
  }
  return _insts[inst];
}

Compiler::Compiler(const std::string &filename, bool singleFunctionMode, bool forceGraphMode)
    : filename_{filename},
      selfManagedParser_{true},
      singleFunctionMode_{singleFunctionMode},
      forceGraphMode_{forceGraphMode},
      walker_{new CompilerNodeVisitor(this)} {
  parser_ = new Parser(filename);
  Init();
}

Compiler::Compiler(Parser *parser, bool singleFunctionMode, bool forceGraphMode)
    : parser_{parser},
      filename_{parser_->filename()},
      selfManagedParser_{false},
      singleFunctionMode_{singleFunctionMode},
      forceGraphMode_{forceGraphMode},
      walker_{new CompilerNodeVisitor(this)} {
  Init();
}

Compiler::~Compiler() {
  if (selfManagedParser_) {
    delete parser_;
  }
  delete walker_;
  LOG_OUT << "Call ~Compiler";
}

void Compiler::Compile() {
  StmtPtr module = parser_->ParseCode();
  if (walker_ == nullptr) {
    LOG_ERROR << "AST walker should not be null.";
    exit(EXIT_FAILURE);
  }
  walker_->Visit(module);
}

bool Compiler::CompileModule(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileExpr(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  CHECK_IF_NULL(stmt->stmt.Expr.value);
  LOG_OUT << ToString(stmt) << "/" << ToString(stmt->stmt.Expr.value);
  CallExprHandler(stmt->stmt.Expr.value);
  const auto lineno = stmt->lineStart;
  InstCall pop = {.inst = Inst_PopTop, .offset = 0, .lineno = lineno};
  AddInstruction(pop);
  return true;
}

bool Compiler::CompileAssign(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Assign) {
    return false;
  }
  // Handle target name.
  const auto &target = stmt->stmt.Assign.target;
  if (target->type != ExprType_Name) {
    LOG_ERROR << "Not a Name, but " << ToString(target);
    exit(EXIT_FAILURE);
  }
  const auto &targetName = *target->expr.Name.identifier;
  // Handle value.
  CallExprHandler(stmt->stmt.Assign.value);
  // Make call.
  const auto lineno = target->lineStart;
  const auto index = FindSymbolIndex(targetName);
  InstCall call = {.inst = Inst_StoreLocal,
                   .offset = (index == -1 ? static_cast<ssize_t>(symbolPool(CurrentCodeIndex()).size()) : index),
                   .lineno = lineno};
  if (index == -1) {  // Not used before.
    symbolPool(CurrentCodeIndex()).emplace_back(targetName);
  }
  LOG_OUT << "name: " << targetName << ", index: " << call.offset;
  AddInstruction(call);
  return true;
}

bool Compiler::CompileAugAssign(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileReturn(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Return) {
    return false;
  }
  // Handle return value.
  const auto &returnVal = stmt->stmt.Return.value;
  if (returnVal != nullptr) {
    CallExprHandler(returnVal);
  }
  // Make return.
  const auto lineno = stmt->lineStart;
  InstCall ret = {.inst = Inst_ReturnVal,
                  .offset = (returnVal != nullptr ? 0 : -1),  // offset is 0, means return value.
                                                              // offset is not 0, means return void.
                  .lineno = lineno};
  AddInstruction(ret);
  return true;
}

bool Compiler::CompileGraph(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Graph) {
    return false;
  }
  const auto &graphStmt = stmt->stmt.Graph;
  // Graph name.
  ExprConstPtr name = graphStmt.name;
  CHECK_IF_NULL(name);
  CHECK_IF_NULL(name->expr.Name.identifier);
  const auto graphName = *(name->expr.Name.identifier);
  const auto lineno = name->lineStart;
  // Insert graph name into the symbol table.
  auto graphSymIndex = FindGlobalSymbolIndex(graphName);
  if (graphSymIndex == -1) {  // Not used before.
    graphSymIndex = symbolPool(0).size();
    symbolPool(0).emplace_back(graphName);
  }
  LOG_OUT << "graph name: " << graphName << ", index: " << graphSymIndex;

  InstCall defineGraph = {.inst = Inst_DefineGraph, .offset = static_cast<ssize_t>(codes().size()), .lineno = lineno};
  AddInstruction(defineGraph);
  Code code{.type = CodeGraph, .name = graphName};
  // Push the graph.
  codeStack_.emplace(codes_.size());
  codes_.emplace_back(std::move(code));
  Code *graphCode = &codes_.back();
  // Graph parameters.
  LOG_OUT << "graph args len: " << graphStmt.argsLen;
  for (size_t i = 0; i < graphStmt.argsLen; ++i) {
    const auto &argStmt = graphStmt.args[i];
    LOG_OUT << "graph args[" << i << "]: " << ToString(argStmt);
    std::string argName;
    if (argStmt->type == StmtType_Expr && argStmt->stmt.Expr.value->type == ExprType_Name) {
      argName = *argStmt->stmt.Expr.value->expr.Name.identifier;
      LOG_OUT << "param: " << argName;
      graphCode->argNames.emplace_back(argName);
      graphCode->argDefaults.emplace_back(Constant());
    } else if (argStmt->type == StmtType_Assign && argStmt->stmt.Assign.target->type == ExprType_Name &&
               argStmt->stmt.Assign.value->type == ExprType_Literal) {
      argName = *argStmt->stmt.Assign.target->expr.Name.identifier;
      const auto &literal = argStmt->stmt.Assign.value->expr.Literal;
      const auto &defaultParam = *literal.value;
      LOG_OUT << "default param: " << argName << ": " << defaultParam;
      graphCode->argNames.emplace_back(argName);
      graphCode->argDefaults.emplace_back(Constant());
    } else {
      CompileMessage(parser_->filename(), lineno, name->columnStart, "error: invalid graph parameters.");
      exit(EXIT_FAILURE);
    }
    // Add graph arguments name at the front of symbol pool.
    auto index = FindSymbolIndex(argName);
    if (index == -1) {  // Not used before.
      LOG_OUT << "arg name: " << argName << ", offset: " << graphCode->symbols.size();
      index = symbolPool(CurrentCodeIndex()).size();
      symbolPool(CurrentCodeIndex()).emplace_back(argName);
    } else {
      CompileMessage(
        parser_->filename(), lineno, name->columnStart,
        "error: invalid graph parameter[" + std::to_string(i) + "]: " + argName + ", already defined before.");
      exit(EXIT_FAILURE);
    }
    graphCode->argIndexes.emplace_back(index);
  }

  // Graph body.
  LOG_OUT << "graph body len: " << graphStmt.len;
  for (size_t i = 0; i < graphStmt.len; ++i) {
    const auto &stmt = graphStmt.body[i];
    CallStmtHandler(stmt);
    LOG_OUT << "graph body[" << i << "]: " << ToString(stmt);
  }
  // Make extra return if no explicit return.
  if (lastInst_.inst != Inst_ReturnVal) {
    InstCall ret = {.inst = Inst_ReturnVal,
                    .offset = -1,  // offset is not 0, means return void.
                    .lineno = lineno};
    AddInstruction(ret);
  }
  codeStack_.pop();

  // Store the graph with name.
  InstCall storeGraph = {.inst = Inst_StoreGlobal, .offset = graphSymIndex, .lineno = lineno};
  AddInstruction(storeGraph);
  return true;
}

void Compiler::CompileJitCallFunction(const std::string &funcName, ssize_t funcSymIndex, int lineno) {
  InstCall loadFunc = {.inst = Inst_LoadGlobal, .offset = funcSymIndex, .lineno = lineno};
  LOG_OUT << "function name: " << funcName << ", index: " << loadFunc.offset;
  AddInstruction(loadFunc);

  // Add JIT arguments here.
  const auto argsLen = 0;
  InstCall call = {.inst = Inst_DoCall,
                   .offset = static_cast<ssize_t>(argsLen),  // Set arguments size as offset.
                   .lineno = lineno};
  AddInstruction(call);
}

bool Compiler::CompileFunction(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Function) {
    return false;
  }
  const auto &funcStmt = stmt->stmt.Function;
  // Function name.
  ExprConstPtr name = funcStmt.name;
  CHECK_IF_NULL(name);
  CHECK_IF_NULL(name->expr.Name.identifier);
  const auto funcName = *(name->expr.Name.identifier);
  const auto lineno = name->lineStart;
  ssize_t funcSymIndex = -1;
  if (!singleFunctionMode_) {
    // Insert function name into the symbol table.
    funcSymIndex = FindGlobalSymbolIndex(funcName);
    if (funcSymIndex == -1) {  // Not used before.
      funcSymIndex = symbolPool(0).size();
      symbolPool(0).emplace_back(funcName);
    }
    LOG_OUT << "funcName: " << funcName << ", index: " << funcSymIndex;

    InstCall defineFunc = {.inst = Inst_DefineFunc, .offset = static_cast<ssize_t>(codes().size()), .lineno = lineno};
    AddInstruction(defineFunc);

    Code code{.type = CodeFunction, .name = funcName};
    // Push the function.
    codeStack_.emplace(codes_.size());
    codes_.emplace_back(std::move(code));
  }

  Code *funcCode = &codes_.back();
  if (singleFunctionMode_) {
    funcCode->name.append(funcName);
  }
  // Function parameters.
  LOG_OUT << "func args len: " << funcStmt.argsLen;
  for (size_t i = 0; i < funcStmt.argsLen; ++i) {
    const auto &argStmt = funcStmt.args[i];
    LOG_OUT << "func args[" << i << "]: " << ToString(argStmt);
    std::string argName;
    if (argStmt->type == StmtType_Expr && argStmt->stmt.Expr.value->type == ExprType_Name) {
      argName = *argStmt->stmt.Expr.value->expr.Name.identifier;
      LOG_OUT << "param: " << argName;
      funcCode->argNames.emplace_back(argName);
      funcCode->argDefaults.emplace_back(Constant());
    } else if (argStmt->type == StmtType_Assign && argStmt->stmt.Assign.target->type == ExprType_Name &&
               argStmt->stmt.Assign.value->type == ExprType_Literal) {
      argName = *argStmt->stmt.Assign.target->expr.Name.identifier;
      const auto &literal = argStmt->stmt.Assign.value->expr.Literal;
      const auto &defaultParam = *literal.value;
      LOG_OUT << "default param: " << argName << ": " << defaultParam;
      funcCode->argNames.emplace_back(argName);
      funcCode->argDefaults.emplace_back(Constant());
    } else {
      CompileMessage(parser_->filename(), lineno, name->columnStart, "error: invalid function parameters.");
      exit(EXIT_FAILURE);
    }
    // Add function arguments name at the front of symbol pool.
    auto index = FindSymbolIndex(argName);
    if (index == -1) {  // Not used before.
      LOG_OUT << "arg name: " << argName << ", offset: " << funcCode->symbols.size();
      index = symbolPool(CurrentCodeIndex()).size();
      symbolPool(CurrentCodeIndex()).emplace_back(argName);
    } else {
      CompileMessage(
        parser_->filename(), lineno, name->columnStart,
        "error: invalid function parameter[" + std::to_string(i) + "]: " + argName + ", already defined before.");
      exit(EXIT_FAILURE);
    }
    funcCode->argIndexes.emplace_back(index);
  }

  // Function body.
  LOG_OUT << "func body len: " << funcStmt.len;
  for (size_t i = 0; i < funcStmt.len; ++i) {
    const auto &stmt = funcStmt.body[i];
    CallStmtHandler(stmt);
    LOG_OUT << "func body[" << i << "]: " << ToString(stmt);
  }
  // Make extra return if no explicit return.
  if (lastInst_.inst != Inst_ReturnVal) {
    InstCall ret = {.inst = Inst_ReturnVal,
                    .offset = -1,  // offset is not 0, means return void.
                    .lineno = lineno};
    AddInstruction(ret);
  }

  if (!singleFunctionMode_) {
    codeStack_.pop();
    // Store the function with name.
    InstCall storeFunc = {.inst = Inst_StoreGlobal, .offset = funcSymIndex, .lineno = lineno};
    AddInstruction(storeFunc);
  }
  return true;
}

bool Compiler::CompileClass(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileBlock(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Block) {
    return false;
  }

#if 0
  // Push the block code.
  auto codeIndex = codes_.size();
  Code code{.type = CodeBlock, .name = "block{#" + std::to_string(codeIndex) + "}"};
  codeStack_.emplace(codes_.size());
  codes_.emplace_back(std::move(code));
#endif

  // Block body.
  const auto &blockStmt = stmt->stmt.Block;
  LOG_OUT << "block body len: " << blockStmt.len;
  for (size_t i = 0; i < blockStmt.len; ++i) {
    const auto &stmt = blockStmt.body[i];
    CallStmtHandler(stmt);
    LOG_OUT << "block body[" << i << "]: " << ToString(stmt);
  }

#if 0
  // Make extra return if no explicit return.
  if (lastInst_.inst != Inst_ReturnVal) {
    InstCall ret = {.inst = Inst_ReturnVal,
                    .offset = -1, // offset is not 0, means return void.
                    .lineno = stmt->lineStart};
    AddInstruction(ret);
  }

  codeStack_.pop();
  InstCall enterBlock = {.inst = Inst_EnterBlock,
                         .offset = static_cast<ssize_t>(codeIndex),
                         .lineno = lineno};
  AddInstruction(enterBlock);
#endif

  return true;
}

bool Compiler::CompileIf(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_If) {
    return false;
  }
  // Handle if condition.
  const auto &cond = stmt->stmt.If.condition;
  CallExprHandler(cond);

  // Create a jump-false branch instruction.
  InstCall jumpFalseInst = {.inst = Inst_JumpFalse,
                            .offset = 0,  // Set as else branch offset later.
                            .lineno = cond->lineStart};
  AddInstruction(jumpFalseInst);
  const auto pendingJumpFalseIndex = CurrentCode().insts.size() - 1;

  // Handle if body.
  const auto ifLen = stmt->stmt.If.ifLen;
  const auto &ifBody = stmt->stmt.If.ifBody;
  for (size_t i = 0; i < ifLen; ++i) {
    CallStmtHandler(ifBody[i]);
  }
  // Add jump instruction for true branch, skip false branch instructions.
  const auto elseLen = stmt->stmt.If.elseLen;
  bool hasLastStmtRetInIfBody = (ifLen != 0 && ifBody[ifLen - 1]->type == StmtType_Return);
  size_t pendingJumpIndex;
  if (elseLen != 0 && !hasLastStmtRetInIfBody) {
    InstCall jumpInst = {.inst = Inst_Jump,
                         .offset = 0,  // Set as if ending offset later.
                         .lineno = cond->lineStart};
    AddInstruction(jumpInst);
    pendingJumpIndex = CurrentCode().insts.size() - 1;
  }
  // Set else offset to jump-false instruction offset.
  CurrentCode().insts[pendingJumpFalseIndex].offset = CurrentCode().insts.size();

  // Handle else body.
  const auto &elseBody = stmt->stmt.If.elseBody;
  for (size_t i = 0; i < elseLen; ++i) {
    CallStmtHandler(elseBody[i]);
  }

  // Set if ending offset to jump instruction offset.
  if (elseLen != 0 && !hasLastStmtRetInIfBody) {
    CurrentCode().insts[pendingJumpIndex].offset = CurrentCode().insts.size();
  }
  return true;
}

bool Compiler::CompileWhile(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_While) {
    return false;
  }
  // Handle if condition.
  const auto condIndex = CurrentCode().insts.size();
  const auto &cond = stmt->stmt.While.condition;
  CallExprHandler(cond);

  // Create a jump-false branch instruction.
  InstCall jumpFalseInst = {.inst = Inst_JumpFalse,
                            .offset = 0,  // Set as the offset after jump-back instruction later.
                            .lineno = cond->lineStart};
  AddInstruction(jumpFalseInst);
  const auto pendingJumpFalseIndex = CurrentCode().insts.size() - 1;

  // Handle if body.
  const auto len = stmt->stmt.While.len;
  const auto &body = stmt->stmt.While.body;
  for (size_t i = 0; i < len; ++i) {
    CallStmtHandler(body[i]);
  }

  // Just jump back to while start position.
  InstCall jumpInst = {.inst = Inst_Jump,
                       .offset = static_cast<ssize_t>(condIndex),  // Set offset as while beginning.
                       .lineno = cond->lineStart};
  AddInstruction(jumpInst);

  // Set jump-false offset just after jump-back instruction.
  CurrentCode().insts[pendingJumpFalseIndex].offset = CurrentCode().insts.size();

  return true;
}

bool Compiler::CompileFor(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileBreak(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileContinue(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompilePass(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileImport(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileStdCin(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_StdCin) {
    return false;
  }
  // Handle stdin variable.
  const auto &in = stmt->stmt.StdCin.value;
  if (in->type != ExprType_Name) {
    return false;
  }
  const auto &name = *in->expr.Name.identifier;
  const auto lineno = in->lineStart;
  const auto index = FindSymbolIndex(name);
  InstCall stdcin = {.inst = Inst_StdCin,
                     .offset = (index == -1 ? static_cast<ssize_t>(symbolPool(CurrentCodeIndex()).size()) : index),
                     .lineno = lineno};
  if (index == -1) {  // Not used before.
    symbolPool(CurrentCodeIndex()).emplace_back(name);
  }
  AddInstruction(stdcin);
  return true;
}

bool Compiler::CompileStdCout(StmtConstPtr stmt) {
  CHECK_IF_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_StdCout) {
    return false;
  }
  // Handle stdout value.
  const auto &stdoutVal = stmt->stmt.StdCout.value;
  CallExprHandler(stdoutVal);
  // Make stdout instruction.
  const auto lineno = stdoutVal->lineStart;
  InstCall stdcout = {.inst = Inst_StdCout, .offset = 0, .lineno = lineno};
  AddInstruction(stdcout);
  return true;
}

bool Compiler::CompileBinary(ExprConstPtr expr) {
  CHECK_IF_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Binary) {
    return false;
  }
  CallExprHandler(expr->expr.Binary.left);
  CallExprHandler(expr->expr.Binary.right);
  const auto lineno = expr->expr.Binary.left->lineStart;
  switch (expr->expr.Binary.op) {
    case OpId_Add: {
      InstCall call = {.inst = Inst_BinaryAdd, .offset = 0, .lineno = lineno};
      AddInstruction(call);
      return true;
    }
    case OpId_Sub: {
      InstCall call = {.inst = Inst_BinarySub, .offset = 0, .lineno = lineno};
      AddInstruction(call);
      return true;
    }
    case OpId_Mul: {
      InstCall call = {.inst = Inst_BinaryMul, .offset = 0, .lineno = lineno};
      AddInstruction(call);
      return true;
    }
    case OpId_Div: {
      InstCall call = {.inst = Inst_BinaryDiv, .offset = 0, .lineno = lineno};
      AddInstruction(call);
      return true;
    }
    case OpId_Equal:
    case OpId_NotEqual:
    case OpId_GreaterThan:
    case OpId_LessThan:
    case OpId_GreaterEqual:
    case OpId_LessEqual: {
      InstCall compare = {.inst = Inst_Compare, .offset = expr->expr.Binary.op, .lineno = lineno};
      AddInstruction(compare);
      return true;
    }
    default:
      break;
  }
  return false;
}

bool Compiler::CompileUnary(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return false;
}

bool Compiler::CompileAttribute(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return false;
}

bool Compiler::CompileSubscript(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return false;
}

bool Compiler::CompileList(ExprConstPtr expr) {
  CHECK_IF_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_List) {
    return false;
  }
  const auto &list = expr->expr.List;
  for (size_t i = 0; i < list.len; ++i) {
    CallExprHandler(list.values[i]);
  }
  return true;
}

bool Compiler::CompileCall(ExprConstPtr expr) {
  CHECK_IF_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Call) {
    return false;
  }

  // Not CallExprHandler(expr->expr.Call.function) to LoadLocal, but LoadGlobal
  // for function name.
  const auto &funcNameExpr = expr->expr.Call.function;
  if (funcNameExpr->type == ExprType_Name) {
    const auto &funcName = *funcNameExpr->expr.Name.identifier;
    const auto lineno = funcNameExpr->lineStart;
    const auto index = FindGlobalSymbolIndex(funcName);
    if (index != -1 && index < intrinsicSize_) {  // Call intrinsic.
      InstCall loadIntrinsic = {.inst = Inst_LoadIntrin, .offset = index, .lineno = lineno};
      LOG_OUT << "intrinsic name: " << funcName << ", index: " << loadIntrinsic.offset << ", size: " << intrinsicSize_;
      AddInstruction(loadIntrinsic);

      CallExprHandler(expr->expr.Call.list);
      const auto argsLen = expr->expr.Call.list->expr.List.len;
      InstCall call = {.inst = Inst_CallIntrin,
                       .offset = static_cast<ssize_t>(argsLen),  // Set arguments size as offset.
                       .lineno = expr->lineStart};
      AddInstruction(call);
      return true;
    } else {  // Call function or graph.
      InstCall loadFunc = {.inst = Inst_LoadGlobal,
                           .offset = (index == -1 ? static_cast<ssize_t>(symbolPool(0).size()) : index),
                           .lineno = lineno};
      if (index == -1) {  // Not used before.
        symbolPool(0).emplace_back(funcName);
      }
      LOG_OUT << "function name: " << funcName << ", index: " << loadFunc.offset;
      AddInstruction(loadFunc);

      CallExprHandler(expr->expr.Call.list);
      const auto argsLen = expr->expr.Call.list->expr.List.len;
      InstCall call = {.inst = Inst_DoCall,
                       .offset = static_cast<ssize_t>(argsLen),  // Set arguments size as offset.
                       .lineno = expr->lineStart};
      AddInstruction(call);
      return true;
    }
  } else if (funcNameExpr->type == ExprType_Attribute) {
    LOG_OUT << "Call attribute, " << ToString(funcNameExpr);
    const auto entity = funcNameExpr->expr.Attribute.entity;
    const auto attr = funcNameExpr->expr.Attribute.attribute;
    if (entity->type == ExprType_Name && attr->type == ExprType_Name) {
      const auto &opsName = *entity->expr.Name.identifier;
      if (opsName == lexer::ToStr(KwId_ops)) {
        // Support ops.xxx for tensor operations.
        const auto &opName = *attr->expr.Name.identifier;
        LOG_OUT << "Call ops." << opName;

        // const auto opSym = opsName + '.' + opName;
        const auto lineno = funcNameExpr->lineStart;
        const auto index = ops::MatchOp(opName.c_str());
        InstCall loadOps = {.inst = Inst_LoadOps, .offset = index, .lineno = lineno};
        LOG_OUT << "Op: " << opName << ", index: " << loadOps.offset;
        AddInstruction(loadOps);

        CallExprHandler(expr->expr.Call.list);
        const auto argsLen = expr->expr.Call.list->expr.List.len;
        InstCall call = {.inst = Inst_CallOps,
                         .offset = static_cast<ssize_t>(argsLen),  // Set arguments size as offset.
                         .lineno = expr->lineStart};
        AddInstruction(call);
        return true;
      }
    }
  }
  return false;
}

bool Compiler::CompileName(ExprConstPtr expr) {
  CHECK_IF_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Name) {
    return false;
  }
  const auto &name = *expr->expr.Name.identifier;
  const auto lineno = expr->lineStart;
  const auto index = FindSymbolIndex(name);
  if (index == -1) {  // Not defined before.
    CompileMessage(parser_->filename(), lineno, expr->columnStart, "error: not defined name: '" + name + "'");
    exit(EXIT_FAILURE);
  }
  InstCall load = {.inst = Inst_LoadLocal,
                   .offset = (index == -1 ? static_cast<ssize_t>(symbolPool(CurrentCodeIndex()).size()) : index),
                   .lineno = lineno};
  LOG_OUT << "name: " << name << ", index: " << load.offset;
  AddInstruction(load);
  return true;
}

bool Compiler::CompileLiteral(ExprConstPtr expr) {
  CHECK_IF_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Literal) {
    return false;
  }
  auto kind = expr->expr.Literal.kind;
  const auto &value = *expr->expr.Literal.value;
  LOG_OUT << "value: " << value;
  const auto lineno = expr->lineStart;
  const auto index = FindConstantIndex(value);
  InstCall load = {.inst = Inst_LoadConst,
                   .offset = index != -1 ? index : static_cast<ssize_t>(constantPool(CurrentCodeIndex()).size()),
                   .lineno = lineno};
  if (index == -1) {  // Not used before.
    Constant cons = {.type = static_cast<ConstType>(kind)};
    cons.value.str = value.c_str();  // Use lexer constant string temporarily, intern it in VM later.
    constantPool(CurrentCodeIndex()).emplace_back(cons);
  }
  AddInstruction(load);
  return true;
}

// Return -1 if not found.
ssize_t Compiler::FindSymbolIndex(const std::string &name) {
  auto &currentSymbolPool = symbolPool(CurrentCodeIndex());
  return FindStringPoolIndex(currentSymbolPool, name);
}

ssize_t Compiler::FindGlobalSymbolIndex(const std::string &name) {
  auto &globalSymbolPool = symbolPool(0);
  auto iter = std::find(globalSymbolPool.cbegin(), globalSymbolPool.cend(), name);
  if (iter == globalSymbolPool.cend()) {
    return -1;
  }
  ssize_t index = std::distance(globalSymbolPool.cbegin(), iter);
  if (index < 0) {
    LOG_ERROR << "Not found symbol, index should not be negative " << index << ", name: " << name;
    exit(EXIT_FAILURE);
  }
  return index;
}

// Return -1 if not found.
ssize_t Compiler::FindConstantIndex(const std::string &str) {
  auto &currentConstantPool = constantPool(CurrentCodeIndex());
  auto iter = std::find_if(currentConstantPool.cbegin(), currentConstantPool.cend(), [&str](const Constant &cons) {
    if (cons.value.str != nullptr && cons.value.str == str.c_str()) {
      return true;
    }
    return false;
  });
  if (iter == currentConstantPool.cend()) {
    return -1;
  }
  ssize_t index = std::distance(currentConstantPool.cbegin(), iter);
  if (index < 0) {
    LOG_ERROR << "Not found constant, index should not be negative " << index << ", str: " << str;
    exit(EXIT_FAILURE);
  }
  return index;
}

void Compiler::Init() {
  InitCompileHandlers();
  codeStack_.emplace(codes_.size());
  if (singleFunctionMode_) {
    if (forceGraphMode_) {
      codes_.emplace_back(Code{.type = CodeGraph, .name = "@single/"});  // Preset module code.
    } else {
      codes_.emplace_back(Code{.type = CodeFunction, .name = "@single/"});  // Preset module code.
    }
  } else {
    codes_.emplace_back(Code{.type = CodeModule, .name = parser_->filename()});  // Preset module code.
  }
  InitIntrinsicSymbols();
}

void Compiler::InitIntrinsicSymbols() {
  // Preset intrinsic symbols in global symbol pool for
  // bool/int/float/str/tensor, and so on.
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_bool));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_int));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_float));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_str));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_list));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_set));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_dict));
  symbolPool(0).emplace_back(lexer::ToStr(LiteralId_tensor));
  symbolPool(0).emplace_back("print");
  intrinsicSize_ = symbolPool(0).size();
}

void Compiler::Dump() {
  std::cout << "--------------------" << std::endl;
  std::cout << "----- bytecode -----" << std::endl;
  std::cout << "total codes: " << codes().size() << std::endl;
  for (size_t codeIndex = 0; codeIndex < codes().size(); ++codeIndex) {
    const auto &code = codes()[codeIndex];
    std::cout << "----------" << std::endl;
    std::cout << "code: <" << ToStr(code.type) << " '" << code.name << "'>";
    std::cout << std::endl;
    if (!code.argNames.empty()) {
      std::cout << "arguments:" << std::endl;
      for (size_t i = 0; i < code.argNames.size(); ++i) {
        const auto &arg = code.argNames[i];
        const auto idx = code.argIndexes[i];
        const auto &def = code.argDefaults[i];
        std::cout << std::setfill(' ') << std::setw(8) << std::left << i;
        if (def.value.str != nullptr && strlen(def.value.str) != 0) {
          std::cout << std::setfill(' ') << std::setw(8) << std::left << arg << ' ' << idx << ' ';
          std::cout << def.value.str;
        } else {
          std::cout << arg << ' ' << idx;
        }
        std::cout << std::endl;
      }
    }

    std::cout << "instructions:" << std::endl;
    ssize_t lastLineno = -1;
    for (size_t i = 0; i < code.insts.size(); ++i) {
      const auto &inst = code.insts[i];
      // Print lineno.
      std::cout << std::setfill(' ') << std::setw(8) << std::left;
      if (lastLineno != inst.lineno) {
        if (lastLineno != -1) {  // Print blank line between lines.
          std::cout << std::endl;
        }
        lastLineno = inst.lineno;
        std::cout << lastLineno;
      } else {
        std::cout << ' ';
      }
      // Print instruction number.
      std::cout << std::setfill(' ') << std::setw(8) << std::left << i;
      // Print instruction.
      std::cout << std::setfill(' ') << std::setw(16) << std::left << GetInstStr(inst.inst);
      // Print variable names or constants.
      switch (inst.inst) {
        case Inst_LoadName:
        case Inst_StoreName:
        case Inst_LoadLocal:
        case Inst_StoreLocal: {
          std::cout << inst.offset << " (" << symbolPool(codeIndex)[inst.offset] << ')';
          break;
        }
        case Inst_LoadGlobal:
        case Inst_StoreGlobal:
        case Inst_LoadIntrin: {
          LOG_OUT << "size: " << symbolPool(0).size() << ", index: " << inst.offset;
          std::cout << inst.offset << " (" << symbolPool(0)[inst.offset] << ')';
          break;
        }
        case Inst_LoadOps: {
          std::cout << inst.offset << " (" << ops::ToStr((ops::Op)inst.offset) << ')';
          break;
        }
        case Inst_StdCin: {
          std::cout << inst.offset << " (" << symbolPool(codeIndex)[inst.offset] << ')';
          break;
        }
        case Inst_JumpTrue:
        case Inst_JumpFalse: {
          std::cout << inst.offset;
          break;
        }
        case Inst_Jump: {
          std::cout << inst.offset;
          break;
        }
        case Inst_Compare: {
          std::cout << inst.offset << " (";
          switch (inst.offset) {
            case OpId_Equal: {
              std::cout << "==" << ')';
              break;
            }
            case OpId_NotEqual: {
              std::cout << "!=" << ')';
              break;
            }
            case OpId_GreaterThan: {
              std::cout << ">" << ')';
              break;
            }
            case OpId_LessThan: {
              std::cout << "<" << ')';
              break;
            }
            case OpId_GreaterEqual: {
              std::cout << ">=" << ')';
              break;
            }
            case OpId_LessEqual: {
              std::cout << "<=" << ')';
              break;
            }
            default: {
              std::cout << "error" << ')';
              break;
            }
          }
          break;
        }
        case Inst_LoadConst: {
          std::cout << inst.offset << " (";
          const auto &cons = constantPool(codeIndex)[inst.offset];
          if (cons.type == ConstType_str) {
            std::cout << "'";
          }
          CHECK_IF_NULL(cons.value.str);
          std::cout << ConvertEscapeString(std::string(cons.value.str));
          if (cons.type == ConstType_str) {
            std::cout << "'";
          }
          std::cout << ')';
          break;
        }
        case Inst_EnterBlock: {
          std::cout << inst.offset;
          break;
        }
        default:
          break;
      }
      std::cout << std::endl;
    }

    std::cout << "symbols:" << std::endl;
    for (size_t i = 0; i < code.symbols.size(); ++i) {
      const auto &var = code.symbols[i];
      std::cout << std::setfill(' ') << std::setw(8) << std::left << i;
      std::cout << var << std::endl;
    }

    std::cout << "constants:" << std::endl;
    for (size_t i = 0; i < code.constants.size(); ++i) {
      const auto &cons = code.constants[i];
      std::cout << std::setfill(' ') << std::setw(8) << std::left << i;
      std::cout << std::setfill(' ') << std::setw(8) << std::left << lexer::ToStr(static_cast<LtId>(cons.type));
      if (cons.type == ConstType_str) {
        std::cout << "'";
      }
      CHECK_IF_NULL(cons.value.str);
      std::cout << ConvertEscapeString(std::string(cons.value.str));
      if (cons.type == ConstType_str) {
        std::cout << "'";
      }
      std::cout << std::endl;
    }
  }
}

#define STMT(type) stmtHandlers_[StmtType_##type] = &Compiler::Compile##type;
#define EXPR(type) exprHandlers_[ExprType_##type] = &Compiler::Compile##type;
void Compiler::InitCompileHandlers() {
#include "lang/c/parser/expr.list"
#include "lang/c/parser/stmt.list"
}
#undef STMT
#undef EXPR
}  // namespace compiler
}  // namespace da
