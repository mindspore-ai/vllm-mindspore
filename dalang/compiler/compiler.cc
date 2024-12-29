#include "compiler/compiler.h"

#undef LOG_OUT
#define LOG_OUT LOG_NO_OUT

#include <algorithm>

#define TO_STR(s) #s
#define INSTRUCTION(I) TO_STR(I),
const char *_insts[]{// Inst strings.
                     "Invalid",
#include "compiler/instruction.list"
                     "End"};
#undef INSTRUCTION

namespace compiler {
const char *GetInstStr(Inst inst) {
  if (inst <= Inst_Invalid || inst >= Inst_End) {
    LOG_ERROR << "inst is abnormal, " << inst << ", " << Inst_End;
  }
  return _insts[inst];
}

Compiler::Compiler(const std::string &filename)
    : parser_{Parser(filename)}, walker_{new CompilerNodeVisitor(this)} {
  InitCompileHandlers();
}

Compiler::~Compiler() { delete walker_; }

void Compiler::Compile() {
  StmtPtr module = parser_.ParseCode();
  if (walker_ == nullptr) {
    LOG_ERROR << "AST walker should not be null.";
    exit(1);
  }
  walker_->Visit(module);
}

bool Compiler::CompileModule(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileExpr(StmtConstPtr stmt) {
  CHECK_NULL(stmt);
  CHECK_NULL(stmt->stmt.Expr.value);
  LOG_OUT << ToString(stmt) << "/" << ToString(stmt->stmt.Expr.value);
  CallExprHandler(stmt->stmt.Expr.value);
  const auto lineno = stmt->lineStart;
  InstCall pop = {.inst = Inst_PopTop, .offset = 0, .lineno = lineno};
  AddInstruction(pop);
  return true;
  return true;
}

bool Compiler::CompileAssign(StmtConstPtr stmt) {
  CHECK_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Assign) {
    return false;
  }
  // Handle target name.
  const auto &target = stmt->stmt.Assign.target;
  if (target->type != ExprType_Name) {
    LOG_ERROR << "Not a Name, but " << ToString(target);
    exit(1);
  }
  const auto &targetName = *target->expr.Name.identifier;
  LOG_OUT << "targetName: " << targetName;
  // Handle value.
  CallExprHandler(stmt->stmt.Assign.value);
  // Make call.
  const auto lineno = target->lineStart;
  const auto index = FindSymbolIndex(targetName);
  InstCall call = {
      .inst = Inst_StoreName,
      .offset =
          (index == -1 ? static_cast<ssize_t>(symbolPool_.size()) : index),
      .lineno = lineno};
  if (index == -1) { // Not used before.
    symbolPool_.emplace_back(targetName);
  }
  AddInstruction(call);
  return true;
}

bool Compiler::CompileAugAssign(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileReturn(StmtConstPtr stmt) {
  CHECK_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Return) {
    return false;
  }
  // Handle return value.
  const auto &returnVal = stmt->stmt.Return.value;
  CallExprHandler(returnVal);
  // Make return.
  const auto lineno = returnVal->lineStart;
  InstCall ret = {.inst = Inst_ReturnVal, .offset = 0, .lineno = lineno};
  AddInstruction(ret);
  return true;
}

bool Compiler::CompileFunction(StmtConstPtr stmt) {
  CHECK_NULL(stmt);
  LOG_OUT << ToString(stmt);
  if (stmt->type != StmtType_Function) {
    return false;
  }
  const auto &funcStmt = stmt->stmt.Function;
  // Function name.
  ExprConstPtr name = funcStmt.name;
  CHECK_NULL(name);
  CHECK_NULL(name->expr.Name.identifier);
  const auto funcName = *(name->expr.Name.identifier);
  LOG_OUT << "func name: " << funcName;
  const auto lineno = name->lineStart;
  const auto index = FindFunctionIndex(funcName);
  if (index != -1) {
    CompileMessage(parser_.filename(), lineno, name->columnStart,
                   "error: duplicate function name '" + funcName + "'.");
    exit(1);
  }
  InstCall funcBegin = {.inst = Inst_FuncBegin,
                        .offset = static_cast<ssize_t>(functionPool_.size()),
                        .lineno = lineno};
  AddInstruction(funcBegin);
  Function func{.name = funcName,
                .offset = static_cast<ssize_t>(instructions_.size())};
  // Function parameters.
  LOG_OUT << "func args len: " << funcStmt.argsLen;
  for (size_t i = 0; i < funcStmt.argsLen; ++i) {
    const auto &argStmt = funcStmt.args[i];
    LOG_OUT << "func args[" << i << "]: " << ToString(argStmt);
    if (argStmt->type == StmtType_Expr &&
        argStmt->stmt.Expr.value->type == ExprType_Name) {
      const auto &argName = *argStmt->stmt.Expr.value->expr.Name.identifier;
      LOG_OUT << "param: " << argName;
      func.args.emplace_back(argName);
      func.defs.emplace_back("");
    } else if (argStmt->type == StmtType_Assign &&
               argStmt->stmt.Assign.target->type == ExprType_Name &&
               argStmt->stmt.Assign.value->type == ExprType_Literal) {
      const auto &argName = *argStmt->stmt.Assign.target->expr.Name.identifier;
      const auto &literal = argStmt->stmt.Assign.value->expr.Literal;
      const auto &defaultParam = *literal.value;
      LOG_OUT << "default param: " << argName << ": " << defaultParam;
      func.args.emplace_back(argName);
      func.defs.emplace_back(defaultParam);
    } else {
      CompileMessage(parser_.filename(), lineno, name->columnStart,
                     "error: invalid function parameters.");
    }
  }
  functionPool_.emplace_back(std::move(func));
  // Also insert function name into the symbol table.
  const auto symIndex = FindSymbolIndex(funcName);
  if (symIndex == -1) { // Not used before.
    symbolPool_.emplace_back(funcName);
  }

  // Function body.
  LOG_OUT << "func body len: " << funcStmt.len;
  for (size_t i = 0; i < funcStmt.len; ++i) {
    const auto &stmt = funcStmt.body[i];
    CallStmtHandler(stmt);
    LOG_OUT << "func body[" << i << "]: " << ToString(stmt);
  }

  InstCall funcEnd = {
      .inst = Inst_FuncEnd, .offset = funcBegin.offset, .lineno = lastLineno_};
  AddInstruction(funcEnd);
  return true;
}

bool Compiler::CompileClass(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileBlock(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileIf(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileWhile(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
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

bool Compiler::CompileBinary(ExprConstPtr expr) {
  CHECK_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Binary) {
    return false;
  }
  CallExprHandler(expr->expr.Binary.left);
  CallExprHandler(expr->expr.Binary.right);
  const auto lineno = expr->expr.Binary.left->lineStart;
  if (expr->expr.Binary.op == OpId_Add) {
    InstCall call = {.inst = Inst_BinaryAdd, .offset = 0, .lineno = lineno};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Sub) {
    InstCall call = {.inst = Inst_BinarySub, .offset = 0, .lineno = lineno};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Mul) {
    InstCall call = {.inst = Inst_BinaryMul, .offset = 0, .lineno = lineno};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Div) {
    InstCall call = {.inst = Inst_BinaryDiv, .offset = 0, .lineno = lineno};
    AddInstruction(call);
    return true;
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
  CHECK_NULL(expr);
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
  CHECK_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Call) {
    return false;
  }
  CallExprHandler(expr->expr.Call.function);
  CallExprHandler(expr->expr.Call.list);
  const auto lineno = expr->lineStart;
  InstCall call = {.inst = Inst_CallFunc, .offset = 0, .lineno = lineno};
  AddInstruction(call);
  return true;
}

bool Compiler::CompileName(ExprConstPtr expr) {
  CHECK_NULL(expr);
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Name) {
    return false;
  }
  const auto &name = *expr->expr.Name.identifier;
  const auto lineno = expr->lineStart;
  const auto index = FindSymbolIndex(name);
  InstCall load = {
      .inst = Inst_LoadName,
      .offset =
          (index == -1 ? static_cast<ssize_t>(symbolPool_.size()) : index),
      .lineno = lineno};
  if (index == -1) { // Not used before.
    symbolPool_.emplace_back(name);
  }
  AddInstruction(load);
  return true;
}

bool Compiler::CompileLiteral(ExprConstPtr expr) {
  CHECK_NULL(expr);
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
                   .offset = index != -1
                                 ? index
                                 : static_cast<ssize_t>(constantPool_.size()),
                   .lineno = lineno};
  if (index == -1) { // Not used before.
    Constant cons = {.type = static_cast<ConstType>(kind), .value = value};
    constantPool_.emplace_back(cons);
  }
  AddInstruction(load);
  return true;
}

// Return -1 if not found.
ssize_t Compiler::FindSymbolIndex(const std::string &name) {
  auto iter = std::find(symbolPool_.cbegin(), symbolPool_.cend(), name);
  if (iter == symbolPool_.cend()) {
    return -1;
  }
  auto index = std::distance(symbolPool_.cbegin(), iter);
  if (index < 0) {
    LOG_ERROR << "Not found symbol, index should not be negative " << index
              << ", name: " << name;
    exit(1);
  }
  return index;
}

// Return -1 if not found.
ssize_t Compiler::FindConstantIndex(const std::string &str) {
  auto iter = std::find_if(constantPool_.cbegin(), constantPool_.cend(),
                           [&str](const Constant &cons) {
                             if (cons.value == str) {
                               return true;
                             }
                             return false;
                           });
  if (iter == constantPool_.cend()) {
    return -1;
  }
  auto index = std::distance(constantPool_.cbegin(), iter);
  if (index < 0) {
    LOG_ERROR << "Not found constant, index should not be negative " << index
              << ", str: " << str;
    exit(1);
  }
  return index;
}

// Return -1 if not found.
ssize_t Compiler::FindFunctionIndex(const std::string &name) {
  auto iter = std::find_if(functionPool_.cbegin(), functionPool_.cend(),
                           [&name](const Function &func) {
                             if (func.name == name) {
                               return true;
                             }
                             return false;
                           });
  if (iter == functionPool_.cend()) {
    return -1;
  }
  auto index = std::distance(functionPool_.cbegin(), iter);
  if (index < 0) {
    LOG_ERROR << "Not found function, index should not be negative " << index
              << ", function: " << name;
    exit(1);
  }
  return index;
}

void Compiler::Dump() {
  std::cout << "instructions: " << std::endl;
  std::cout << "-----" << std::endl;
  ssize_t lastLineno = -1;
  for (size_t i = 0; i < instructions_.size(); ++i) {
    const auto &inst = instructions_[i];
    // Print lineno.
    if (lastLineno != inst.lineno) {
      if (lastLineno != -1) { // Print blank line between lines.
        std::cout << std::endl;
      }
      lastLineno = inst.lineno;
      std::cout << lastLineno;
    }
    // Print instruction number.
    std::cout << '\t' << std::setw(8) << std::right << i << ' ';
    // Print instruction.
    std::cout << GetInstStr(inst.inst);
    // Print variable names or constants.
    if (inst.inst == Inst_LoadName || inst.inst == Inst_StoreName) {
      std::cout << "\t\t" << inst.offset << " (" << symbolPool_[inst.offset]
                << ')';
    } else if (inst.inst == Inst_FuncBegin || inst.inst == Inst_FuncEnd) {
      std::cout << "\t\t" << inst.offset << " ("
                << functionPool_[inst.offset].name << ')';
    } else if (inst.inst == Inst_LoadConst) {
      std::cout << "\t\t" << inst.offset << " (";
      const auto &cons = constantPool_[inst.offset];
      if (cons.type == ConstType_str) {
        std::cout << "'";
      }
      std::cout << cons.value;
      if (cons.type == ConstType_str) {
        std::cout << "'";
      }
      std::cout << ')';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "symbols: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < symbolPool_.size(); ++i) {
    const auto &var = symbolPool_[i];
    std::cout << i << "\t\t" << var << std::endl;
  }
  std::cout << std::endl;

  std::cout << "constants: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < constantPool_.size(); ++i) {
    const auto &cons = constantPool_[i];
    std::cout << i << "\t\t";
    if (cons.type == ConstType_str) {
      std::cout << "'";
    }
    std::cout << cons.value;
    if (cons.type == ConstType_str) {
      std::cout << "'";
    }
    std::cout << '\t' << ToStr(static_cast<LtId>(cons.type)) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "functions: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < functionPool_.size(); ++i) {
    const auto &func = functionPool_[i];
    std::cout << i << "\t\t";

    std::cout << func.name << "(";
    if (func.args.size() != func.defs.size()) {
      std::cout << "\n\nerror: parameters count is not equal to defaults count."
                << std::endl;
      exit(1);
    }
    for (size_t i = 0; i < func.args.size(); ++i) {
      const auto &arg = func.args[i];
      const auto &def = func.defs[i];
      std::cout << arg;
      if (!def.empty()) {
        std::cout << '=' << def;
      }
      if (i != func.args.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << ")" << std::endl;
  }
  std::cout << std::endl;
}

#define STMT(type) stmtHandlers_[StmtType_##type] = &Compiler::Compile##type;
#define EXPR(type) exprHandlers_[ExprType_##type] = &Compiler::Compile##type;
void Compiler::InitCompileHandlers() {
#include "parser/expr.list"
#include "parser/stmt.list"
}
#undef STMT
#undef EXPR
} // namespace compiler