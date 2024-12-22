
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
} // namespace

bool VM::ReplaceEscapeStr(std::string &dst) {
  constexpr auto escapeSize = 4;
  const char *escapes[] = {"\\\\", "\\n", "\\r", "\\t"};
  const char *results[] = {"\\", "\n", "\r", "\t"};
  std::string::size_type pos = 0;
  for (EVER) {
    pos = dst.find('\\', pos);
    if (pos == std::string::npos) {
      return true;
    }
    if (pos + 1 == dst.size()) { // Meet a last '\'
      return false;
    }
    std::string::size_type oldPos = pos;
    for (size_t i = 0; i < escapeSize; ++i) {
      auto escChar = escapes[i][1];
      if (dst[pos + 1] == escChar) {
        dst.replace(pos, strlen(escapes[i]), results[i]);
        LOG_OUT << "replace " << escapes[i] << " with " << results[i];
        pos += strlen(results[i]);
        break;
      }
    }
    if (oldPos == pos) { // Not match any escape.
      return false;
    }
  }
  return true;
}

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
    const char *strPtr = stringPool().Intern(value);
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
  stack().emplace_back(ConvertConstType(cons));
}

// Load a value by name which stored before.
void VM::InstLoadName(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &name = syms()[offset];
  auto iter = names().find(name);
  if (iter == names().cend()) {
    CompileMessage(LineString(), "error: not defined symbol: '" + name + "'");
    exit(1);
  }
  stack().emplace_back(iter->second);
}

// Store a slot by name for latish load.
void VM::InstStoreName(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &name = syms()[offset];
  auto iter = names().find(name);
  if (iter != names().cend()) {
    CompileMessage(LineString(), "warning: covered symbol: '" + name + "'");
  }
  names()[name] = std::move(stack().back());
  stack().pop_back();
}

// Just pop up the top slot.
void VM::InstPopTop(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  stack().pop_back();
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
    LOG_OUT << "stack size: " << stack().size();
  }

  // Print the value if Return exists.
  if (frames_.size() == 1 && stack().size() == 1 &&
      currentInstPtr_->inst == Inst_ReturnVal) {
    std::cout << ToString(stack().back());
  }
}

#define INSTRUCTION(I) instHandlers_[Inst_##I] = &VM::Inst##I;
void VM::InitInstructionHandlers() {
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm