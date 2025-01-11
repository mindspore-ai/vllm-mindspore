
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
  const auto &cons = consts()[offset];
  LOG_OUT << "offset: " << offset << ", value: " << cons.value << " ("
          << cons.type << ")";
  CurrentStack().emplace_back(ConvertConstType(cons));
}

// Load a value by name which stored before.
void VM::InstLoadName(ssize_t offset) {
  const auto &name = syms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  // Find the name from current frame and upper frames, one frame by one frame.
  auto *slot = FindLoadedName(name);
  if (slot == nullptr) {
    // Not found in all namespaces.
    CompileMessage(LineString(), "error: not defined symbol: '" + name + "'.");
    exit(1);
  }
  LOG_OUT << "load: " << ToString(*slot);
  CurrentStack().emplace_back(*slot);
}

// Store a slot by name for latish load.
void VM::InstStoreName(ssize_t offset) {
  if (CurrentStack().empty()) {
    CompileMessage(LineString(), "error: stack is empty.");
    exit(1);
  }
  const auto &name = syms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  auto iter = names().find(name);
  if (iter == names().cend()) {
    // First declared name in current frame.
    LOG_OUT << "no defined symbol: '" + name + "' in current frame.";
    Slot *slot = FindLoadedName(name);
    if (slot == nullptr) {
      // Create a new variable and store into it.
      names()[name] = std::move(CurrentStack().back());
    } else {
      // Store the value into upper frame's names.
      *slot = std::move(CurrentStack().back());
    }
  } else {
    // Already used name.
    names()[name] = std::move(CurrentStack().back());
  }
  CurrentStack().pop_back();
}

// Just pop up all the slots.
void VM::InstPopTop(ssize_t offset) {
  LOG_OUT << "offset: " << offset << ", return value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  CurrentStack().clear();
}

BINARY_OP(Add, +) // VM::InstBinaryAdd
BINARY_OP(Sub, -) // VM::InstBinarySub
BINARY_OP(Mul, *) // VM::InstBinaryMul
BINARY_OP(Div, /) // VM::InstBinaryDiv

// Call a function by function slot and argument slots, and create a new frame.
// The first(reverse stack) slot is function object created by DefineFunc
// instruction. The left slot is arguments.
void VM::InstCallFunc(ssize_t offset) {
  const auto &funcNameSlot = CurrentStack().front();
  if (funcNameSlot.type != SlotFunction || CurrentStack().size() < 1) {
    CompileMessage(
        compiler_->filename(), 0, 0,
        "error: invalid function name. slot: " + ToString(funcNameSlot) +
            ", stack size: " + std::to_string(CurrentStack().size()));
    exit(1);
  }

  // Get callee function information.
  const auto codeIndex = funcNameSlot.value.offset;
  const auto &func = codes()[codeIndex];
  LOG_OUT << "offset: " << offset << ", name: " << func.name;
  // Create new function frame in advance.
  auto newFuncFrame =
      Frame{.type = FrameFunction, .code = funcNameSlot.value.offset};

  if (CurrentStack().size() > 1) { // Has arguments.
    // To bind the arguments and parameters.
    auto argsSize = CurrentStack().size() - 1;
    auto paramsSize = func.args.size();
    if (argsSize > paramsSize) {
      std::stringstream ss;
      ss << "error: function arguments size(" << argsSize
         << ") should not exceed parameters size(" << paramsSize << ").";
      CompileMessage(compiler_->filename(), 0, 0, ss.str());
      exit(1);
    }

    // Move all arguments from caller stack into callee names map.
    for (size_t i = 0; i < argsSize; ++i) {
      auto argIndex = i + 1; // Not include function name;
      const auto &arg = CurrentStack()[argIndex];
      newFuncFrame.names[func.args[i]] = std::move(arg);
    }
    CurrentStack().clear();

    // Append default parameters in callee names map.
    if (argsSize < paramsSize) {
      // for (size_t i = argsSize; i < paramsSize; ++i) {
      //   names()[func.args[i]] = std::move(Slot{.type=SlotInt,
      //   .value.int_=func.defs[i]});
      // }
      LOG_ERROR << "Not support default parameter by now";
    }
  }

  // Push a new frame for function call.
  frames_.emplace_back(newFuncFrame);
}

// Return the top slot to previous frame.
//   offset (0): explicit return set by user.
//   offset (not 0): the compiler append it for every function implicitly, if no
//   return set by user.
void VM::InstReturnVal(ssize_t offset) {
  const auto &slot = CurrentStack().front();
  LOG_OUT << "offset: " << offset << ", return value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  if (frames_.size() <= 1) {
    CompileMessage(compiler_->filename(), 0, 0,
                   "error: only one frame left, can not return anymore.");
    exit(1);
  }
  // If explicit return, move the value into previous frame stack.
  if (offset == 0) {
    auto &prevFrame = frames_.rbegin()[1];
    prevFrame.slots.emplace_back(std::move(slot));
  }
  // Just pop the frame.
  frames_.pop_back();
}

// Make a function object slot and push into the stack.
// Usually, a StoreName instruction is followed, to bind a function name.
void VM::InstDefineFunc(ssize_t offset) {
  const auto &func = codes()[offset];
  LOG_OUT << "offset: " << offset << ", function: " << func.name;
  auto iter = names().find(func.name);
  if (iter != names().cend()) {
    CompileMessage(LineString(),
                   "error: redefined function symbol: '" + func.name + "'.");
    exit(1);
  }
  Slot funcSlot{.type = SlotFunction};
  funcSlot.value.offset = offset;
  CurrentStack().emplace_back(std::move(funcSlot));
}

// Push a new frame for block.
// The code of block can be accessed by offset.
void VM::InstEnterBlock(ssize_t offset) {
  const auto &block = codes()[offset];
  LOG_OUT << "offset: " << offset << ", block: " << block.name;

  // Create new frame for block.
  auto blockFrame =
      Frame{.type = FrameBlock, .code = static_cast<size_t>(offset)};
  // Push a new frame for function call.
  frames_.emplace_back(blockFrame);
}

// Input from standard input stream.
void VM::InstStdCin(ssize_t offset) {
  Slot *slot = nullptr;
  const auto &name = syms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  auto iter = names().find(name);
  if (iter == names().cend()) {
    // First declared name in current frame.
    LOG_OUT << "no defined symbol: '" + name + "' in current frame.";
    slot = FindLoadedName(name);
    if (slot == nullptr) {
      // Create a new variable and store into it.
      slot = &names()[name];
    }
  } else {
    // Already used name.
    slot = &names()[name];
  }

  // Get input.
  static std::string str;
  getline(std::cin, str);
  CHECK_NULL(slot);
  slot->type = SlotString;
  slot->value.str_ = str.c_str();
}

// Output to standard output stream.
void VM::InstStdCout(ssize_t offset) {
  LOG_OUT << "offset: " << offset << ", std out value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  if (CurrentStack().size() < 1) {
    CompileMessage(compiler_->filename(), 0, 0,
                   "error: no slot left, can not output by stdout.");
    exit(1);
  }
  // Print the value.
  const auto &slot = CurrentStack().front();
  std::cout << ToString(slot);
  CurrentStack().pop_back();
}

// Return nullptr if not found.
Slot *VM::FindLoadedName(const std::string &str) {
  for (auto iter = frames_.rbegin(); iter != frames_.rend(); ++iter) {
    auto nameIter = iter->names.find(str);
    if (nameIter != iter->names.end()) {
      return &nameIter->second;
    }
  }
  return nullptr;
}

void VM::Run() {
  auto topFrame = Frame{.type = FrameModule, .code = 0};
  frames_.emplace_back(topFrame);
  while (!frames_.empty()) {
    // Run in current frame.
    while (CurrentPc() < insts().size()) {
      const auto &inst = insts()[CurrentPc()];
      if (inst.inst >= instHandlers_.size()) {
        LOG_ERROR << "instruction handler list size is less than input "
                     "inst type: "
                  << inst.inst << " >= " << instHandlers_.size();
        exit(1);
      }
      ++CurrentPc();

      // Print the value if Return exists in the module.
      if (frames_.size() == 1 && CurrentStack().size() == 1 &&
          inst.inst == Inst_ReturnVal && inst.offset == 0) {
        std::cout << ToString(CurrentStack().back());
        break;
      }

      (this->*instHandlers_[inst.inst])(inst.offset);
      LOG_OUT << "frame size: " << frames_.size()
              << ", stack size: " << CurrentStack().size()
              << ", inst size: " << insts().size() << ", pc: " << CurrentPc();
    }
    // Pop exhausted frame.
    frames_.pop_back();
  }
}

#define INSTRUCTION(I) instHandlers_[Inst_##I] = &VM::Inst##I;
void VM::InitInstructionHandlers() {
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm