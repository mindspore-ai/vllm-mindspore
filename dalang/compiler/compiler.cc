#include "compiler/compiler.h"

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
  LOG_OUT << ToString(stmt) << "/" << ToString(stmt->stmt.Expr.value);
  return false;
}

bool Compiler::CompileAssign(StmtConstPtr stmt) {
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
  const auto targetName = *target->expr.Name.identifier;
  LOG_OUT << "targetName: " << targetName;
  // Handle value.
  CallExprHandler(stmt->stmt.Assign.value);
  // Make call.
  InstCall call = {.inst = Inst_StoreName,
                   .offset = static_cast<ssize_t>(variablePool_.size())};
  variablePool_.emplace_back(targetName);
  AddInstruction(call);
  return true;
}

bool Compiler::CompileAugAssign(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileReturn(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
}

bool Compiler::CompileFunction(StmtConstPtr stmt) {
  LOG_OUT << ToString(stmt);
  return false;
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
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Binary) {
    return false;
  }
  CallExprHandler(expr->expr.Binary.left);
  CallExprHandler(expr->expr.Binary.right);
  if (expr->expr.Binary.op == OpId_Add) {
    InstCall call = {.inst = Inst_BinaryAdd, .offset = 0};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Sub) {
    InstCall call = {.inst = Inst_BinarySub, .offset = 0};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Mul) {
    InstCall call = {.inst = Inst_BinaryMul, .offset = 0};
    AddInstruction(call);
    return true;
  } else if (expr->expr.Binary.op == OpId_Div) {
    InstCall call = {.inst = Inst_BinaryDiv, .offset = 0};
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
  LOG_OUT << ToString(expr);
  return false;
}

bool Compiler::CompileCall(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  return false;
}

bool Compiler::CompileName(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Name) {
    return false;
  }
  const auto &name = *expr->expr.Name.identifier;
  auto iter = std::find(variablePool_.cbegin(), variablePool_.cend(), name);
  auto index = std::distance(variablePool_.cbegin(), iter);
  InstCall load = {.inst = Inst_LoadName, .offset = index};
  AddInstruction(load);
  return true;
}

bool Compiler::CompileLiteral(ExprConstPtr expr) {
  LOG_OUT << ToString(expr);
  if (expr->type != ExprType_Literal) {
    return false;
  }
  auto kind = expr->expr.Literal.kind;
  const auto &value = *expr->expr.Literal.value;
  LOG_OUT << "value: " << value;
  InstCall load = {.inst = Inst_LoadConst,
                   .offset = static_cast<ssize_t>(constantPool_.size())};
  Constant cons = {.type = static_cast<ConstType>(kind), .value = value};
  constantPool_.emplace_back(cons);
  AddInstruction(load);
  return true;
}

void Compiler::Dump() {
  std::cout << "instructions: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < instructions_.size(); ++i) {
    const auto &inst = instructions_[i];
    std::cout << '#' << i << ":\t" << GetInstStr(inst.inst);
    if (inst.inst == Inst_LoadName || inst.inst == Inst_StoreName) {
      std::cout << "\t" << inst.offset << " (" << variablePool_[inst.offset]
                << ')';
    } else if (inst.inst == Inst_LoadConst) {
      std::cout << "\t" << inst.offset << " (";
      std::cout << constantPool_[inst.offset].value << ')';
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
  std::cout << "variables: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < variablePool_.size(); ++i) {
    const auto &var = variablePool_[i];
    std::cout << '#' << i << ":\t" << var << std::endl;
  }
  std::cout << std::endl;
  std::cout << "constants: " << std::endl;
  std::cout << "-----" << std::endl;
  for (size_t i = 0; i < constantPool_.size(); ++i) {
    const auto &cons = constantPool_[i];
    std::cout << '#' << i << ":\t" << cons.value << '\t'
              << ToStr(static_cast<LtId>(cons.type)) << std::endl;
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