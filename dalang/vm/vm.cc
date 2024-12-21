
#include "vm/vm.h"

#include "common/common.h"
#include <algorithm>

#undef LOG_OUT
#define LOG_OUT LOG_NO_OUT

namespace vm {
namespace {
void ReplaceStr(std::string &dst, const char *oldStr, size_t oldStrLen,
                const char *newStr) {
  std::string::size_type pos = 0;
  while ((pos = dst.find(oldStr)) != std::string::npos) {
    dst.replace(pos, oldStrLen, newStr);
  }
}

void ConvertEscapeChar(std::string &str) {
  ReplaceStr(str, "\\n", 2, "\n");
  ReplaceStr(str, "\\r", 2, "\r");
  ReplaceStr(str, "\\t", 2, "\t");
}
} // namespace

Slot VM::ConvertConstType(ConstType type, const std::string &value) {
  Slot slot;
  switch (type) {
  case ConstType_bool: {
    slot.type = SlotBool;
    slot.value.bool_ = value == "true" ? true : false;
    return slot;
  }
  case ConstType_int: {
    slot.type = SlotInt;
    slot.value.int_ = std::stoi(value);
    return slot;
  }
  case ConstType_float: {
    slot.type = SlotFloat;
    slot.value.float_ = std::stof(value);
    return slot;
  }
  case ConstType_str: {
    slot.type = SlotString;
    std::string val = value;
    ConvertEscapeChar(val);
    const char *strPtr = stringPool().Intern(std::move(val));
    slot.value.str_ = strPtr;
    return slot;
  }
  default:
    throw std::runtime_error("Unexcepted constant");
  }
}

Slot VM::ConvertConstType(const Constant &cons) {
  return ConvertConstType(cons.type, cons.value);
}

// Load the constants by index from const pool.
void VM::InstLoadConst(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &cons = consts()[offset];
  stack_.emplace_back(ConvertConstType(cons));
}

// Load a value by name which stored before.
void VM::InstLoadName(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &name = syms()[offset];
  auto iter = nameMap_.find(name);
  if (iter == nameMap_.cend()) {
    CompileMessage(LineString(), "error: not defined symbol: '" + name + "'");
    exit(1);
  }
  stack_.emplace_back(iter->second);
}

// Store a slot by name for latish load
void VM::InstStoreName(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &name = syms()[offset];
  auto iter = nameMap_.find(name);
  if (iter != nameMap_.cend()) {
    CompileMessage(LineString(), "warning: covered symbol: '" + name + "'");
  }
  nameMap_[name] = std::move(stack_.back());
  stack_.pop_back();
}

BINARY_OP(Add, +) // VM::InstBinaryAdd
BINARY_OP(Sub, -) // VM::InstBinarySub
BINARY_OP(Mul, *) // VM::InstBinaryMul
BINARY_OP(Div, /) // VM::InstBinaryDiv

void VM::InstCallFunc(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstReturnVal(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstFuncBegin(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstFuncEnd(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstClassBegin(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstClassEnd(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::Run() {
  for (const auto &inst : insts()) {
    if (inst.inst >= instHandlers_.size()) {
      LOG_ERROR << "instruction handler list size is less than input "
                   "inst type.";
      exit(1);
    }
    currentInstPtr_ = &inst;
    (this->*instHandlers_[inst.inst])(inst.offset);
    LOG_OUT << "stack size: " << stack_.size();
  }

  // Print the value if Return exists.
  if (stack_.size() == 1 && currentInstPtr_->inst == Inst_ReturnVal) {
    std::cout << ToString(stack_.back());
  }
}

#define INSTRUCTION(I) instHandlers_[Inst_##I] = &VM::Inst##I;
void VM::InitInstructionHandlers() {
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm