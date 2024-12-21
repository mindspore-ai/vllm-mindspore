#ifndef __VM_VM_H__
#define __VM_VM_H__

#include <iomanip>
#include <limits>
#include <set>

#include "compiler/compiler.h"

namespace vm {
using namespace compiler;

enum SlotType {
  SlotInvalid,
  SlotRefName,
  SlotFunction,
  SlotClass,
  SlotBool,
  SlotInt,
  SlotFloat,
  SlotString,
  SlotMax
};

struct Slot {
  SlotType type;
  union {
    void *addr;
    bool bool_;
    ssize_t int_;
    double float_;
    const char *str_;
  } value;
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
  VM(Compiler *compiler)
      : compiler_{compiler}, instsPtr_{&compiler->instructions()},
        symsPtr_{&compiler->symbolPool()},
        constsPtr_{&compiler->constantPool()} {
    InitInstructionHandlers();
  }

  void InstLoadConst(ssize_t offset);
  void InstLoadName(ssize_t offset);
  void InstStoreName(ssize_t offset);
  void InstBinaryAdd(ssize_t offset);
  void InstBinarySub(ssize_t offset);
  void InstBinaryMul(ssize_t offset);
  void InstBinaryDiv(ssize_t offset);
  void InstCallFunc(ssize_t offset);
  void InstReturnVal(ssize_t offset);
  void InstFuncBegin(ssize_t offset);
  void InstFuncEnd(ssize_t offset);
  void InstClassBegin(ssize_t offset);
  void InstClassEnd(ssize_t offset);

  void Run();

  StringPool &stringPool() { return stringPool_; }

private:
  const std::vector<InstCall> &insts() const {
    CHECK_NULL(instsPtr_);
    return *instsPtr_;
  }

  const std::vector<std::string> &syms() const {
    CHECK_NULL(symsPtr_);
    return *symsPtr_;
  }

  const std::vector<Constant> &consts() const {
    CHECK_NULL(constsPtr_);
    return *constsPtr_;
  }

  void InitInstructionHandlers();

  std::string LineString() {
    return compiler_->filename() + ':' +
           std::to_string(currentInstPtr_->lineno);
  }

  Slot ConvertConstType(ConstType type, const std::string &value);
  Slot ConvertConstType(const Constant &cons);

  bool ReplaceEscapeStr(std::string &dst);

  Compiler *compiler_{nullptr};
  const std::vector<InstCall> *instsPtr_{nullptr};
  const std::vector<std::string> *symsPtr_{nullptr};
  const std::vector<Constant> *constsPtr_{nullptr};

  StringPool stringPool_;

  std::vector<Slot> stack_;
  std::unordered_map<std::string, Slot> nameMap_;

  using InstHandlerFunction = void (VM::*)(ssize_t);
  std::unordered_map<InstType, InstHandlerFunction> instHandlers_;
  const InstCall *currentInstPtr_{nullptr};
};

#define BINARY_OP(OpName, OpSymbol)                                            \
  void VM::InstBinary##OpName(ssize_t offset) {                                \
    LOG_OUT << "offset: " << offset;                                           \
    const auto &rhs = std::move(stack_.back());                                \
    stack_.pop_back();                                                         \
    const auto &lhs = std::move(stack_.back());                                \
    stack_.pop_back();                                                         \
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
      stack_.emplace_back(std::move(slot));                                    \
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
      stack_.emplace_back(std::move(slot));                                    \
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
      stack_.emplace_back(slot);                                               \
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
      stack_.emplace_back(std::move(slot));                                    \
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
      stack_.emplace_back(std::move(slot));                                    \
    } else {                                                                   \
      CompileMessage(LineString(),                                             \
                     "error: only support int or float binary operation.");    \
      exit(1);                                                                 \
    }                                                                          \
  }

} // namespace vm
#endif // __VM_VM_H__