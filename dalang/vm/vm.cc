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

#include "vm/vm.h"

#include "common/common.h"
#include <algorithm>

#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT

#undef DEBUG

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
    exit(EXIT_FAILURE);
  }
  LOG_OUT << "load: " << ToString(*slot);
  CurrentStack().emplace_back(*slot);
}

// Store a slot by name for latish load.
void VM::InstStoreName(ssize_t offset) {
  if (CurrentStack().empty()) {
    CompileMessage(LineString(), "error: stack is empty.");
    exit(EXIT_FAILURE);
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
  // Found the function slot firstly.
  auto funcIter =
      std::find_if(CurrentStack().crbegin(), CurrentStack().crend(),
                   [](const Slot &slot) { return slot.type == SlotFunction; });
  if (funcIter == CurrentStack().crend() || CurrentStack().size() < 1) {
    CompileMessage(
        filename(), 0, 0,
        "error: invalid function name. slot: " + ToString(*funcIter) +
            ", stack size: " + std::to_string(CurrentStack().size()));
    exit(EXIT_FAILURE);
  }
  const auto &funcNameSlot = *funcIter;
  const auto argsSize =
      static_cast<size_t>(std::distance(CurrentStack().crbegin(), funcIter));

  // Get callee function information.
  const auto codeIndex = funcNameSlot.value.offset;
  const auto &func = codes()[codeIndex];
  LOG_OUT << "offset: " << offset << ", name: " << func.name
          << ", id: " << frames_.size();
  // Create new function frame in advance.
  auto newFuncFrame =
      Frame{.type = FrameFunction, .code = funcNameSlot.value.offset};

  if (argsSize > 0) { // Has arguments.
    // To bind the arguments and parameters.
    auto paramsSize = func.args.size();
    if (argsSize > paramsSize) {
      std::stringstream ss;
      ss << "error: function arguments size(" << argsSize
         << ") should not exceed parameters size(" << paramsSize << ").";
      CompileMessage(filename(), 0, 0, ss.str());
      exit(EXIT_FAILURE);
    }

    // Move all arguments from caller stack into callee names map.
    auto argStartIndex = CurrentStack().size() - argsSize;
    for (size_t i = 0; i < argsSize; ++i) {
      const auto &arg = CurrentStack()[argStartIndex + i];
      newFuncFrame.names[func.args[i]] = std::move(arg);
    }
    // Erase all arguments.
    CurrentStack().erase(CurrentStack().begin() + argStartIndex,
                         CurrentStack().end());

    // Append default parameters in callee names map.
    if (argsSize < paramsSize) {
      // for (size_t i = argsSize; i < paramsSize; ++i) {
      //   names()[func.args[i]] = std::move(Slot{.type=SlotInt,
      //   .value.int_=func.defs[i]});
      // }
      LOG_ERROR << "Not support default parameter by now";
    }
  }
  // Erase the function.
  CurrentStack().pop_back();

  // Push a new frame for function call.
  frames_.emplace_back(newFuncFrame);
}

// Return the top slot to previous frame.
//   offset (0): explicit return set by user.
//   offset (not 0): the compiler append it for every function implicitly, if no
//   return set by user.
void VM::InstReturnVal(ssize_t offset) {
  if (frames_.size() < 1) {
    CompileMessage(filename(), 0, 0,
                   "error: no frame left, can not return anymore.");
    exit(EXIT_FAILURE);
  }
  LOG_OUT << "offset: " << offset
          << ", return from function, name: " << code().name
          << ", id: " << (frames_.size() - 1) << ", value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  // If explicit return, move the value into previous frame stack.
  if (offset == 0) {
    auto &prevFrame = frames_.rbegin()[1];
    const auto &slot = CurrentStack().back();
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
    exit(EXIT_FAILURE);
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

namespace {
COMPARE_OP(==)
COMPARE_OP(!=)
COMPARE_OP(>)
COMPARE_OP(<)
COMPARE_OP(>=)
COMPARE_OP(<=)
} // namespace

void VM::InstCompare(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &rhs = std::move(CurrentStack().back());
  CurrentStack().pop_back();
  const auto &lhs = std::move(CurrentStack().back());
  CurrentStack().pop_back();

  bool res;
  if (offset == OpId_Equal) {
    res = lhs == rhs;
  } else if (offset == OpId_NotEqual) {
    res = lhs != rhs;
  } else if (offset == OpId_GreaterThan) {
    res = lhs > rhs;
  } else if (offset == OpId_LessThan) {
    res = lhs < rhs;
  } else if (offset == OpId_GreaterEqual) {
    res = lhs >= rhs;
  } else if (offset == OpId_LessEqual) {
    res = lhs <= rhs;
  }

  Slot slot = {.type = SlotBool};
  slot.value.bool_ = res;
  LOG_OUT << "condition: " << res;
  CurrentStack().emplace_back(std::move(slot));
}

// Change the pc to offset if the condition is true.
void VM::InstJumpTrue(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &slot = CurrentStack().back();
  if (slot.type != SlotBool) {
    CompileMessage(LineString(), "error: the condition type is not bool: '" +
                                     ToString(slot) + "'.");
    exit(EXIT_FAILURE);
  }
  if (slot.value.bool_) {
    LOG_OUT << "Jump from " << CurrentPc() << " to " << offset;
    CurrentPc() = offset;
  }
  CurrentStack().pop_back();
}

// Change the pc to offset if the condition is false.
void VM::InstJumpFalse(ssize_t offset) {
  LOG_OUT << "offset: " << offset;
  const auto &slot = CurrentStack().back();
  if (slot.type != SlotBool) {
    CompileMessage(LineString(), "error: the condition type is not bool: '" +
                                     ToString(slot) + "'.");
    exit(EXIT_FAILURE);
  }
  LOG_OUT << "condition: " << slot.value.bool_;
  if (!slot.value.bool_) {
    LOG_OUT << "Jump from " << CurrentPc() << " to " << offset;
    CurrentPc() = offset;
  }
  CurrentStack().pop_back();
}

// Change the pc unconditionally.
void VM::InstJump(ssize_t offset) {
  LOG_OUT << "offset: " << offset;

  LOG_OUT << "Jump from " << CurrentPc() << " to " << offset;
  CurrentPc() = offset;
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
  std::string str;
  getline(std::cin, str);
  CHECK_NULL(slot);
  slot->type = SlotString;
  const char *strPtr = stringPool().Intern(std::move(str));
  slot->value.str_ = strPtr;
}

// Output to standard output stream.
void VM::InstStdCout(ssize_t offset) {
  LOG_OUT << "offset: " << offset << ", std out value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  if (CurrentStack().size() < 1) {
    CompileMessage(filename(), 0, 0,
                   "error: no slot left, can not output by stdout.");
    exit(EXIT_FAILURE);
  }
  // Print the value.
  const auto &slot = CurrentStack().back();
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

size_t &VM::CurrentPc() { return frames_.back().pc; }

std::vector<Slot> &VM::CurrentStack() { return frames_.back().slots; }

std::unordered_map<std::string, Slot> &VM::names() {
  return frames_.back().names;
}

StringPool &VM::stringPool() { return stringPool_; }

const std::vector<Code> &VM::codes() const {
  CHECK_NULL(codesPtr_);
  return *codesPtr_;
}

const Code &VM::code() const { return codes()[frames_.back().code]; }

const std::vector<std::string> &VM::syms() const {
  return codes()[frames_.back().code].symbols;
}

const std::vector<Constant> &VM::consts() const {
  return codes()[frames_.back().code].constants;
}

const std::vector<InstCall> &VM::insts() const {
  return codes()[frames_.back().code].insts;
}

const std::string &VM::filename() const { return filename_; }

std::string VM::LineString() {
  return filename() + ':' + std::to_string(insts()[CurrentPc() - 1].lineno);
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
        exit(EXIT_FAILURE);
      }
      ++CurrentPc();

      // Print the value if Return exists in the module.
      if (frames_.size() == 1 && CurrentStack().size() == 1 &&
          inst.inst == Inst_ReturnVal && inst.offset == 0) {
        std::cout << ToString(CurrentStack().back());
        break;
      }

      (this->*instHandlers_[inst.inst])(inst.offset);
      if (frames_.empty()) {
        // Finish.
        LOG_OUT << "Run finish";
        return;
      }
      LOG_OUT << "frame size: " << frames_.size()
              << ", stack size: " << CurrentStack().size()
              << ", inst size: " << insts().size() << ", pc: " << CurrentPc();
#ifdef DEBUG
      DumpStack();
#endif
    }
    // Pop exhausted frame.
    frames_.pop_back();
  }
}

void VM::DumpStack() {
  std::cout << "----------" << std::endl;
  std::cout << "frame:" << std::endl;
  for (size_t i = 0; i < frames_.size(); ++i) {
    std::cout << "\t#" << i << ": " << codes()[frames_[i].code].name
              << std::endl;
  }
  std::cout << "stack:" << std::endl;
  for (size_t i = 0; i < CurrentStack().size(); ++i) {
    const auto &slot = CurrentStack()[i];
    std::cout << "\t#" << i << ": " << ToString(slot) << std::endl;
  }
}

#define INSTRUCTION(I) instHandlers_[Inst_##I] = &VM::Inst##I;
void VM::InitInstructionHandlers() {
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm