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

#include <algorithm>
#include <functional>

#include "common/common.h"
#include "vm/intrinsic.h"

#undef DEBUG
#ifndef DEBUG
#undef LOG_OUT
#define LOG_OUT NO_LOG_OUT
#endif

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
  case ConstType_tensor: {
    slot.type = SlotTensor;
    slot.value.tensor_ =
        nullptr; // TODO: convert string value to tensor value later.
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
  CurrentStack().emplace_back(std::move(ConvertConstType(cons)));
}

// Load a value by name which stored before.
void VM::InstLoadName(ssize_t offset) {
  const auto &name = LocalSyms()[offset];
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
    CompileMessage(LineString(), "error: stack is empty.\nfail to store name.");
    exit(EXIT_FAILURE);
  }
  const auto &name = LocalSyms()[offset];
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

// Load a value by offset which stored before.
void VM::InstLoadLocal(ssize_t offset) {
  const auto &name = LocalSyms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  auto *slot = &LocalVars()[offset];
  LOG_OUT << "load: " << ToString(*slot);
  CurrentStack().emplace_back(*slot);
}

// Store a slot by offset for latish load.
void VM::InstStoreLocal(ssize_t offset) {
  if (CurrentStack().empty()) {
    CompileMessage(LineString(),
                   "error: stack is empty.\nfail to store local.");
    exit(EXIT_FAILURE);
  }
  LocalVars()[offset] = std::move(CurrentStack().back());
  CurrentStack().pop_back();
}

// Load a value by offset which stored before.
void VM::InstLoadGlobal(ssize_t offset) {
  const auto &name = GlobalSyms()[offset];
  LOG_OUT << "offset: " << offset << "/" << GlobalVars().size()
          << ", name: " << name;
  auto *slot = &GlobalVars()[offset];
  LOG_OUT << "load: " << ToString(*slot);
  CurrentStack().emplace_back(*slot);
}

// Store a slot by offset for latish load.
void VM::InstStoreGlobal(ssize_t offset) {
  const auto &name = GlobalSyms()[offset];
  LOG_OUT << "offset: " << offset << ", name: " << name;
  if (CurrentStack().empty()) {
    CompileMessage(LineString(),
                   "error: stack is empty.\nfail to store global.");
    exit(EXIT_FAILURE);
  }
  GlobalVars()[offset] = std::move(CurrentStack().back());
  CurrentStack().pop_back();
}

// Load an intrinsic by offset.
void VM::InstLoadIntrin(ssize_t offset) {
  const auto &name = GlobalSyms()[offset];
  LOG_OUT << "offset: " << offset << "/" << GlobalVars().size()
          << ", name: " << name;
  Slot intrinSlot{.type = SlotIntrinsic};
  switch (offset) {
  case LiteralId_bool: {
    break;
  }
  case LiteralId_int: {
    break;
  }
  case LiteralId_float: {
    break;
  }
  case LiteralId_str: {
    break;
  }
  case LiteralId_list: {
    break;
  }
  case LiteralId_set: {
    break;
  }
  case LiteralId_dict: {
    break;
  }
  case LiteralId_tensor: {
    intrinSlot.value.intr = intrinsic::IntrisicType_tensor;
    break;
  }
  default:
    break;
  }

  CurrentStack().emplace_back(std::move(intrinSlot));
}

// Load an ops by offset.
void VM::InstLoadOps(ssize_t offset) {
  const auto &name = ops::ToStr((ops::Op)offset);
  LOG_OUT << "offset: " << offset << ", name: " << name;
  Slot opsSlot{.type = SlotOps};
  opsSlot.value.op = (ops::Op)offset;
  CurrentStack().emplace_back(std::move(opsSlot));
}

// Just pop up the top slot.
void VM::InstPopTop(ssize_t offset) {
  LOG_OUT << "offset: " << offset << ", return value: "
          << (CurrentStack().empty() ? "<null>"
                                     : ToString(CurrentStack().back()))
          << ", stack size: " << CurrentStack().size();
  if (CurrentStack().empty()) {
    CompileMessage(LineString(), "error: stack is empty.\nfail to pop top.");
    exit(EXIT_FAILURE);
  }
  CurrentStack().pop_back();
}

BINARY_OP(Add, +) // VM::InstBinaryAdd
BINARY_OP(Sub, -) // VM::InstBinarySub
BINARY_OP(Mul, *) // VM::InstBinaryMul
BINARY_OP(Div, /) // VM::InstBinaryDiv

// Call a function or graph by function/graph slot and argument slots, and
// create a new frame. The first(reverse stack) slot is function/graph object
// created by DefineFunc/DefineGraph instruction. The left slot is arguments.
void VM::InstDoCall(ssize_t offset) {
  if (CurrentStack().size() < 1) {
    CompileMessage(filename(), 0, 0,
                   "error: invalid function. stack size: " +
                       std::to_string(CurrentStack().size()));
    exit(EXIT_FAILURE);
  }
#if 0
  // Found the function slot firstly.
  auto funcIter =
      std::find_if(CurrentStack().crbegin(), CurrentStack().crend(),
                   [](const Slot &slot) { return slot.type == SlotFunction; });
  if (funcIter == CurrentStack().crend()) {
    CompileMessage(filename(), 0, 0,
                   "error: not found function slot. stack size: " +
                       std::to_string(CurrentStack().size()));
    exit(EXIT_FAILURE);
  }
  const auto &funcNameSlot = *funcIter;
  const auto argsSize =
      static_cast<size_t>(std::distance(CurrentStack().crbegin(), funcIter));
#else
  const auto argsSize = static_cast<size_t>(offset);
  const auto &funcNameSlot = *(CurrentStack().crbegin() + argsSize);
#endif

  // Get callee function/graph information.
  const auto codeIndex = funcNameSlot.value.offset;
  const auto &callCode = codes()[codeIndex];
  LOG_OUT << "offset: " << offset << ", type: " << callCode.type
          << ", name: " << callCode.name << ", id: " << frames_.size();

  // If call a graph.
  if (callCode.type == CodeGraph) {
    std::cout << "#DAGraph: " << callCode.name << std::endl;
    graphExecutor_.BeginGraph();
  }

  // Create new function/graph frame in advance.
  auto newFuncFrame = Frame{.type = callCode.type,
                            .code = funcNameSlot.value.offset,
                            .vars = std::vector<Slot>{callCode.symbols.size()}};

  if (argsSize > 0) { // Has arguments.
    // To bind the arguments and parameters.
    auto paramsSize = callCode.args.size();
    if (argsSize > paramsSize) {
      std::stringstream ss;
      ss << "error: ";
      ss << (callCode.type == CodeGraph ? "graph" : "function");
      ss << " arguments size(" << argsSize
         << ") should not exceed parameters size(" << paramsSize << ").";
      CompileMessage(filename(), 0, 0, ss.str());
      exit(EXIT_FAILURE);
    }

    // Move all arguments from caller stack into callee names map.
    auto argStartIndex = CurrentStack().size() - argsSize;
    for (size_t i = 0; i < argsSize; ++i) {
      const auto &arg = CurrentStack()[argStartIndex + i];
#if 0
      newFuncFrame.names[callCode.args[i]] = std::move(arg);
#else
      LOG_OUT << "vars offset: " << i << ", name: " << code().symbols[i]
              << ", arg: " << ToString(arg) << ", the same as StoreLocal.";
      newFuncFrame.vars[i] =
          std::move(arg); // The argument name must be at the front.
#endif
    }
    // Erase all arguments.
    CurrentStack().erase(CurrentStack().begin() + argStartIndex,
                         CurrentStack().end());

    // Append default parameters in callee names map.
    if (argsSize < paramsSize) {
      // for (size_t i = argsSize; i < paramsSize; ++i) {
      //   names()[callCode.args[i]] = std::move(Slot{.type=SlotInt,
      //   .value.int_=callCode.defs[i]});
      // }
      LOG_ERROR << "Not support default parameter by now";
    }
  }
  // Erase the function/graph.
  CurrentStack().pop_back();

  // Push a new frame for function/graph call.
  frames_.emplace_back(std::move(newFuncFrame));
  frame_ = &frames_.back();
}

// Call an intrinsic.
void VM::InstCallIntrin(ssize_t offset) {
  if (CurrentStack().size() < 1) {
    CompileMessage(filename(), 0, 0,
                   "error: invalid intrinsic call. stack size: " +
                       std::to_string(CurrentStack().size()));
    exit(EXIT_FAILURE);
  }
  const auto argsSize = static_cast<size_t>(offset);
  const auto &intrinsicSlot = *(CurrentStack().crbegin() + argsSize);

  switch (intrinsicSlot.value.intr) {
  case intrinsic::IntrisicType_bool: {
    break;
  }
  case intrinsic::IntrisicType_int: {
    break;
  }
  case intrinsic::IntrisicType_float: {
    break;
  }
  case intrinsic::IntrisicType_str: {
    break;
  }
  case intrinsic::IntrisicType_list: {
    break;
  }
  case intrinsic::IntrisicType_set: {
    break;
  }
  case intrinsic::IntrisicType_dict: {
    break;
  }
  case intrinsic::IntrisicType_tensor: {
    auto tensor = graphExecutor_.AddTensor();
    Slot tensorSlot{.type = SlotTensor};
    tensorSlot.value.addr = tensor;
    CurrentStack().emplace_back(std::move(tensorSlot));
    break;
  }
  default:
    LOG_ERROR << "invalid intrinsic.";
    exit(EXIT_FAILURE);
  }

  if (argsSize > 0) { // Has arguments.
    // Erase all arguments.
    auto argStartIndex = CurrentStack().size() - argsSize;
    CurrentStack().erase(CurrentStack().begin() + argStartIndex,
                         CurrentStack().end());
  }
  // Erase the intrinsic self.
  CurrentStack().pop_back();
  LOG_OUT << "Call intrisic.";
}

// Call an ops.xxx.
void VM::InstCallOps(ssize_t offset) {
  if (CurrentStack().size() < 1) {
    CompileMessage(filename(), 0, 0,
                   "error: invalid ops call. stack size: " +
                       std::to_string(CurrentStack().size()));
    exit(EXIT_FAILURE);
  }

  const auto argsSize = static_cast<size_t>(offset);
  const auto &opsNameSlot = *(CurrentStack().crbegin() + argsSize);

  // Call an op.
  std::cout << "  #call ops." << ops::ToStr(opsNameSlot.value.op) << std::endl;

  if (argsSize > 0) { // Has arguments.
    // Erase all arguments.
    auto argStartIndex = CurrentStack().size() - argsSize;
    CurrentStack().erase(CurrentStack().begin() + argStartIndex,
                         CurrentStack().end());
  }
  // Erase the op self.
  CurrentStack().pop_back();

  // TODO: Remove later, just for test.
  LOG_OUT << "Call ops";
  Slot fakeSlot{.type = SlotInt};
  fakeSlot.value.int_ = -1;
  CurrentStack().emplace_back(std::move(fakeSlot));
}

// Return the top slot to previous frame.
//   offset (0): explicit return set by user.
//   offset (not 0): the compiler append it for every function implicitly, if
//   no return set by user.
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

  if (offset == 0) {
    // If explicit return value, move the value into previous frame stack.
    const auto &slot = CurrentStack().back();
    auto &prevFrame = frames_.rbegin()[1];
    prevFrame.slots.emplace_back(std::move(slot));
  } else {
    // Set a void slot for later pop top instruction.
    Slot voidSlot = {.type = SlotVoid};
    auto &prevFrame = frames_.rbegin()[1];
    prevFrame.slots.emplace_back(std::move(voidSlot));
  }
  // Just pop the frame.
  frames_.pop_back();
  frame_ = &frames_.back();
}

// Make a function object slot and push into the stack.
// Usually, a StoreName instruction is followed, to bind a function name.
void VM::InstDefineFunc(ssize_t offset) {
  const auto &func = codes()[offset];
  LOG_OUT << "offset: " << offset << ", function: " << func.name;
  // auto iter = names().find(func.name);
  // if (iter != names().cend()) {
  //   CompileMessage(LineString(),
  //                  "error: redefined function symbol: '" + func.name +
  //                  "'.");
  //   exit(EXIT_FAILURE);
  // }
  Slot funcSlot{.type = SlotFunction};
  funcSlot.value.offset = offset;
  CurrentStack().emplace_back(std::move(funcSlot));
}

// Make a graph object slot and push into the stack.
// Usually, a StoreName instruction is followed, to bind a graph name.
void VM::InstDefineGraph(ssize_t offset) {
  const auto &graph = codes()[offset];
  LOG_OUT << "offset: " << offset << ", graph: " << graph.name;
  Slot graphSlot{.type = SlotGraph};
  graphSlot.value.offset = offset;
  CurrentStack().emplace_back(std::move(graphSlot));
}

// Push a new frame for block.
// The code of block can be accessed by offset.
void VM::InstEnterBlock(ssize_t offset) {
  const auto &block = codes()[offset];
  LOG_OUT << "offset: " << offset << ", block: " << block.name;

#if 0
  // Create new frame for block.
  auto blockFrame = Frame{.type = CodeBlock,
                          .code = static_cast<size_t>(offset),
                          .vars = std::vector<Slot>{block.symbols.size()}};
  // Push a new frame for function call.
  frames_.emplace_back(std::move(blockFrame));
  frame_ = &frames_.back();
#endif
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
  try {
    switch (offset) {
    case OpId_Equal: {
      res = lhs == rhs;
      break;
    }
    case OpId_NotEqual: {
      res = lhs != rhs;
      break;
    }
    case OpId_GreaterThan: {
      res = lhs > rhs;
      break;
    }
    case OpId_LessThan: {
      res = lhs < rhs;
      break;
    }
    case OpId_GreaterEqual: {
      res = lhs >= rhs;
      break;
    }
    case OpId_LessEqual: {
      res = lhs <= rhs;
      break;
    }
    default:
      break;
    }
  } catch (std::runtime_error &ex) {
    CompileMessage(LineString(),
                   std::string("error: not support to do [") +
                       ToStr((OpId)offset) + "] compare between '" +
                       GetSlotTypeStr(lhs) + "' and '" + GetSlotTypeStr(rhs) +
                       "'. {" + ToString(lhs) + ", " + ToString(rhs) + "}.");
    exit(EXIT_FAILURE);
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
    LOG_OUT << "Jump from " << frame_->pc << " to " << offset;
    frame_->pc = offset;
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
    LOG_OUT << "Jump from " << frame_->pc << " to " << offset;
    frame_->pc = offset;
  }
  CurrentStack().pop_back();
}

// Change the pc unconditionally.
void VM::InstJump(ssize_t offset) {
  LOG_OUT << "offset: " << offset;

  LOG_OUT << "Jump from " << frame_->pc << " to " << offset;
  frame_->pc = offset;
}

// Input from standard input stream.
void VM::InstStdCin(ssize_t offset) {
#if 0
  Slot *slot = nullptr;
  const auto &name = LocalSyms()[offset];
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
#else
  auto *slot = &LocalVars()[offset];
#endif

  // Get input.
  std::string str;
  getline(std::cin, str);
  CHECK_NULL(slot);
  if ((str.front() == '\'' && str.back() == '\'') ||
      (str.front() == '\"' && str.back() == '\"')) {
    slot->type = SlotString;
    const char *strPtr = stringPool().Intern(str.substr(1, str.size() - 2));
    slot->value.str_ = strPtr;
  } else if (str.find('.') != std::string::npos) {
    slot->type = SlotFloat;
    try {
      slot->value.float_ = std::stod(str);
    } catch (std::invalid_argument &ex) {
      CompileMessage(LineString(),
                     "error: invalid input for float type: " + str);
      exit(EXIT_FAILURE);
    } catch (std::out_of_range &ex) {
      CompileMessage(LineString(), "error: out of range as float type: " + str);
      exit(EXIT_FAILURE);
    }
  } else {
    slot->type = SlotInt;
    try {
      slot->value.int_ = std::stoi(str);
    } catch (std::invalid_argument &ex) {
      CompileMessage(LineString(), "error: invalid input for int type: " + str);
      exit(EXIT_FAILURE);
    } catch (std::out_of_range &ex) {
      CompileMessage(LineString(), "error: out of range as int type: " + str);
      exit(EXIT_FAILURE);
    }
  }
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

std::vector<Slot> &VM::CurrentStack() { return frame_->slots; }

std::unordered_map<std::string, Slot> &VM::names() { return frame_->names; }

std::vector<Slot> &VM::LocalVars() { return frame_->vars; }
std::vector<Slot> &VM::GlobalVars() { return frames_.front().vars; }

StringPool &VM::stringPool() { return stringPool_; }

const std::vector<Code> &VM::codes() const {
  CHECK_NULL(codesPtr_);
  return *codesPtr_;
}

const Code &VM::code() const { return codes()[frame_->code]; }

const std::vector<std::string> &VM::LocalSyms() const {
  return codes()[frame_->code].symbols;
}

const std::vector<std::string> &VM::GlobalSyms() const {
  return codes()[frames_.front().code].symbols;
}

const std::vector<Constant> &VM::consts() const {
  return codes()[frame_->code].constants;
}

const std::vector<InstCall> &VM::insts() const {
  return codes()[frame_->code].insts;
}

const std::string &VM::filename() const { return filename_; }

std::string VM::LineString() {
  return filename() + ':' + std::to_string(insts()[frame_->pc - 1].lineno);
}

void VM::Run() {
  if (codes().empty()) {
    LOG_ERROR << "no code exits";
    exit(EXIT_FAILURE);
  }
  auto topFrame = Frame{.type = CodeModule,
                        .code = 0,
                        .vars = std::vector<Slot>{codes()[0].symbols.size()}};
  frames_.emplace_back(std::move(topFrame));
  while (!frames_.empty()) {
    // Run in current frame.
    frame_ = &frames_.back();
    while (frame_->pc < insts().size()) {
      const auto &inst = insts()[frame_->pc];
      if (inst.inst >= instHandlers_.size()) {
        LOG_ERROR << "instruction handler list size is less than input "
                     "inst type: "
                  << inst.inst << " >= " << instHandlers_.size();
        exit(EXIT_FAILURE);
      }
      ++frame_->pc;

      if (frames_.size() == 1 && CurrentStack().size() == 1 &&
          inst.inst == Inst_ReturnVal && inst.offset == 0) {
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
              << ", inst size: " << insts().size() << ", pc: " << frame_->pc;
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
  instHandlers_ = std::vector<InstHandlerFunction>(Inst_End);
#include "compiler/instruction.list"
}
#undef INSTRUCTION
} // namespace vm