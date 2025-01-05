#ifndef __VM_VM_H__
#define __VM_VM_H__

#include <iomanip>
#include <limits>
#include <set>

#include "compiler/compiler.h"

namespace vm {
using namespace compiler;

class VM;
using InstHandlerFunction = void (VM::*)(ssize_t);
using InstHandlerFunctions = std::unordered_map<InstType, InstHandlerFunction>;

enum SlotType {
  SlotRefName,
  SlotFunction,
  SlotClass,
  SlotBool,
  SlotInt,
  SlotFloat,
  SlotString,
  SlotEnd
};

struct Slot {
  SlotType type;
  union {
    void *addr;
    size_t offset;
    bool bool_;
    ssize_t int_;
    double float_;
    const char *str_;
  } value;
};

enum FrameType { FrameBlock, FrameFunction, FrameModule, FrameEnd };

struct Frame {
  FrameType type;
  size_t code;
  size_t pc{0};
  std::vector<Slot> slots;                     // Slot stack.
  std::unordered_map<std::string, Slot> names; // Name map.
};

inline void GetSlotStr(const Slot &slot, std::stringstream &ss) {
  switch (slot.type) {
  case SlotBool: {
    ss << (slot.value.bool_ ? "true" : "false");
    break;
  }
  case SlotInt: {
    ss << std::to_string(slot.value.int_);
    break;
  }
  case SlotFloat: {
    ss << std::setprecision(std::numeric_limits<double>::max_digits10)
       << slot.value.float_;
    break;
  }
  case SlotString: {
    ss << slot.value.str_;
    break;
  }
  case SlotFunction: {
    ss << "function:" << slot.value.addr;
    break;
  }
  case SlotClass: {
    ss << "class:" << slot.value.addr;
    break;
  }
  case SlotRefName: {
    ss << "ref:" << slot.value.addr;
    break;
  }
  default:
    // unknown
    ss << "<unknown>";
  }
}

inline std::string ToString(const Slot &slot) {
  std::stringstream ss;
  GetSlotStr(slot, ss);
  return ss.str();
}

class StringPool {
public:
  const char *Intern(const char *str) {
    return stringPool_.emplace(str).first->c_str();
  }

  const char *Intern(const std::string &str) {
    return stringPool_.emplace(str).first->c_str();
  }
  const char *Intern(const std::string &&str) {
    return stringPool_.emplace(std::move(str)).first->c_str();
  }

private:
  std::set<std::string> stringPool_;
};

class VM {
public:
  VM() = delete;
  VM(Compiler *compiler) : compiler_{compiler}, codesPtr_{&compiler->codes()} {
    InitInstructionHandlers();
  }
  virtual ~VM() = default;

  void Run();

private:
  void InstLoadConst(ssize_t offset);
  void InstLoadName(ssize_t offset);
  void InstStoreName(ssize_t offset);
  void InstPopTop(ssize_t offset);
  void InstBinaryAdd(ssize_t offset);
  void InstBinarySub(ssize_t offset);
  void InstBinaryMul(ssize_t offset);
  void InstBinaryDiv(ssize_t offset);
  void InstCallFunc(ssize_t offset);
  void InstReturnVal(ssize_t offset);
  void InstDefineFunc(ssize_t offset);
  void InstEnterBlock(ssize_t offset);

  size_t &CurrentPc() { return frames_.back().pc; }
  std::vector<Slot> &CurrentStack() { return frames_.back().slots; }
  std::unordered_map<std::string, Slot> &names() {
    return frames_.back().names;
  }

  StringPool &stringPool() { return stringPool_; }

  const std::vector<Code> &codes() const {
    CHECK_NULL(codesPtr_);
    return *codesPtr_;
  }

  const Code &code() const { return codes()[frames_.back().code]; }

  const std::vector<std::string> &syms() const {
    return codes()[frames_.back().code].symbols;
  }

  const std::vector<Constant> &consts() const {
    return codes()[frames_.back().code].constants;
  }

  const std::vector<InstCall> &insts() const {
    return codes()[frames_.back().code].insts;
  }

  Slot *FindLoadedName(const std::string &str);
  bool SetLoadedName(const std::string &str, Slot &&slot);

  void InitInstructionHandlers();

  std::string LineString() {
    return compiler_->filename() + ':' +
           std::to_string(insts()[CurrentPc() - 1].lineno);
  }

  Slot ConvertConstType(ConstType type, const std::string &value);
  Slot ConvertConstType(const Constant &cons);

  bool ReplaceEscapeStr(std::string &dst);

  bool SkipFuncDefine(const InstCall &inst, size_t &funcDefDepth);

  Compiler *compiler_{nullptr};

  const std::vector<Code> *codesPtr_{nullptr};

  StringPool stringPool_;

  std::vector<Frame> frames_; // Block, function or module stack.

  InstHandlerFunctions instHandlers_; // Notice: Do not change.
};

#define BINARY_OP(OpName, OpSymbol)                                            \
  void VM::InstBinary##OpName(ssize_t offset) {                                \
    LOG_OUT << "offset: " << offset;                                           \
    const auto &rhs = std::move(CurrentStack().back());                        \
    CurrentStack().pop_back();                                                 \
    const auto &lhs = std::move(CurrentStack().back());                        \
    CurrentStack().pop_back();                                                 \
    if (lhs.type == SlotInt && rhs.type == SlotInt) {                          \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.int_;                                            \
      if (rhsVal == 0 && strcmp(#OpSymbol, "/") == 0) {                        \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(1);                                                               \
      }                                                                        \
      auto res = lhsVal OpSymbol rhsVal;                                       \
      Slot slot;                                                               \
      slot.type = SlotInt;                                                     \
      slot.value.int_ = res;                                                   \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotFloat && rhs.type == SlotFloat) {               \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.float_;                                          \
      if (rhsVal == 0.0 && strcmp(#OpSymbol, "/") == 0) {                      \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(1);                                                               \
      }                                                                        \
      auto res = lhsVal OpSymbol rhsVal;                                       \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotInt && rhs.type == SlotFloat) {                 \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.float_;                                          \
      if (rhsVal == 0.0 && strcmp(#OpSymbol, "/") == 0) {                      \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(1);                                                               \
      }                                                                        \
      auto res = ((double)lhsVal)OpSymbol rhsVal;                              \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      CurrentStack().emplace_back(slot);                                       \
    } else if (lhs.type == SlotFloat && rhs.type == SlotInt) {                 \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.int_;                                            \
      if (rhsVal == 0 && strcmp(#OpSymbol, "/") == 0) {                        \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(1);                                                               \
      }                                                                        \
      auto res = lhsVal OpSymbol((double)rhsVal);                              \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotString || rhs.type == SlotString) {             \
      if (strcmp(#OpSymbol, "+") != 0) {                                       \
        CompileMessage(LineString(),                                           \
                       "error: only support '+' for string operation.");       \
        exit(1);                                                               \
      }                                                                        \
      std::stringstream ss;                                                    \
      GetSlotStr(lhs, ss);                                                     \
      GetSlotStr(rhs, ss);                                                     \
      Slot slot;                                                               \
      slot.type = SlotString;                                                  \
      const char *str = stringPool().Intern(std::move(ss.str()));              \
      slot.value.str_ = str;                                                   \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else {                                                                   \
      CompileMessage(LineString(),                                             \
                     "error: only support int or float binary operation.");    \
      exit(1);                                                                 \
    }                                                                          \
  }
} // namespace vm
#endif // __VM_VM_H__