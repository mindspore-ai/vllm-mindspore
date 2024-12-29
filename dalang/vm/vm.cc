
#include "vm/vm.h"

#include "common/common.h"
#include <algorithm>

// #undef LOG_OUT
// #define LOG_OUT LOG_NO_OUT

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
  const auto &cons = consts()[offset];
  LOG_OUT << "offset: " << offset << ", value: " << cons.value << " ("
          << cons.type << ")";
  stack().emplace_back(ConvertConstType(cons));
}

// Load a value by name which stored before.
void VM::InstLoadName(ssize_t offset) {
  const auto &name = syms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  auto iter = names().find(name);
  if (iter == names().cend()) {
    CompileMessage(LineString(), "error: not defined symbol: '" + name + "'.");
    exit(1);
  }
  stack().emplace_back(iter->second);
}

// Store a slot by name for latish load.
void VM::InstStoreName(ssize_t offset) {
  const auto &name = syms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  auto iter = names().find(name);
  if (iter != names().cend()) {
    CompileMessage(LineString(), "warning: covered symbol: '" + name + "'.");
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

void VM::InstCallFunc(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &funcName = stack().front();
  if (funcName.type != SlotFunction) {
    CompileMessage(compiler_->filename(), 0, 0,
                   "error: invalid function name. slot: " + ToString(funcName));
    exit(1);
  }
  const auto &func = funcs()[funcName.value.offset];
  // Bind the arguments and parameters.
  auto argsSize = stack().size() - 1;
  auto paramsSize = func.args.size();
  if (argsSize > paramsSize) {
    CompileMessage(
        compiler_->filename(), 0, 0,
        "error: function arguments size should not exceed parameters size.");
  }
  for (size_t i = 0; i < argsSize; ++i) {
    auto argIndex = i + 1; // Not include function name;
    const auto &arg = stack()[argIndex];
    names()[func.args[i]] = std::move(arg);
  }
  if (argsSize < paramsSize) {
    // for (size_t i = argsSize; i < paramsSize; ++i) {
    //   names()[func.args[i]] = std::move(Slot{.type=SlotInt,
    //   .value.int_=func.defs[i]});
    // }
    LOG_ERROR << "Not support default parameter by now";
  }

  stack().clear();
  pc_ = func.offset;
}

void VM::InstReturnVal(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstFuncBegin(ssize_t offset) {
  const auto &func = funcs()[offset];
  LOG_OUT << "offset: " << offset << ", function: " << func.name;
  auto iter = names().find(func.name);
  if (iter != names().cend()) {
    CompileMessage(LineString(),
                   "error: redefined function symbol: '" + func.name + "'.");
    exit(1);
  }
  Slot funcSlot{.type = SlotFunction};
  funcSlot.value.offset = offset;
  names()[func.name] = std::move(funcSlot);
}

void VM::InstFuncEnd(ssize_t offset) {
  const auto &func = funcs()[offset];
  LOG_OUT << "offset: " << offset << ", function: " << func.name;
}

void VM::InstClassBegin(ssize_t offset) { LOG_OUT << "offset: " << offset; }

void VM::InstClassEnd(ssize_t offset) { LOG_OUT << "offset: " << offset; }

bool VM::SkipFuncDefine(const InstCall &inst, size_t &funcDefDepth) {
  if (inst.inst == Inst_FuncBegin) {
    InstFuncBegin(inst.offset);
    ++funcDefDepth;
    return true;
  }
  if (inst.inst == Inst_FuncEnd) {
    if (funcDefDepth == 0) {
      CompileMessage(LineString(),
                     "error: invalid function definition. no function begin?");
      exit(1);
    }
    InstFuncEnd(inst.offset);
    --funcDefDepth;
    return true;
  }
  if (funcDefDepth > 0) {
    return true;
  }
  return false;
}

void VM::Run() {
  size_t funcDefDepth = 0;
  size_t offset = 0;
  while (offset < insts().size()) {
    const auto &inst = insts()[offset];
    if (inst.inst >= instHandlers_.size()) {
      LOG_ERROR << "instruction handler list size is less than input "
                   "inst type.";
      exit(1);
    }
    // Jump the function definition.
    if (SkipFuncDefine(inst, funcDefDepth)) {
      ++offset;
      LOG_OUT << "skip func def, stack size: " << stack().size()
              << ", offset: " << offset;
      continue;
    }

    pc_ = offset;
    (this->*instHandlers_[inst.inst])(inst.offset);
    if (pc_ == offset) {
      ++offset;
    } else {
      offset = pc_;
    }
    LOG_OUT << "stack size: " << stack().size() << ", offset: " << offset;
  }

  // Print the value if Return exists.
  if (frames_.size() == 1 && stack().size() == 1 &&
      insts()[pc_].inst == Inst_ReturnVal) {
    std::cout << ToString(stack().back());
  }
}

#define INSTRUCTION(I) instHandlers_[Inst_##I] = &VM::Inst##I;
void VM::InitInstructionHandlers() {
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm