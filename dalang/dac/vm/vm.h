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

#ifndef __VM_VM_H__
#define __VM_VM_H__

#include <iomanip>
#include <limits>
#include <set>

#include "compiler/compiler.h"
#include "ops/ops_name.h"
#include "runtime/executor.h"
#include "vm/intrinsic.h"

namespace da {
namespace vm {
using namespace compiler;

class VM;
using InstHandlerFunction = void (VM::*)(ssize_t);
using InstHandlerFunctions = std::vector<InstHandlerFunction>;

enum SlotType {
  SlotInvalid,

  // Function, graph or class.
  SlotFunction,
  SlotGraph,
  SlotClass,

  // Data type.
  SlotBool,
  SlotInt,
  SlotFloat,
  SlotString,
  SlotTensor,

  // Extension type.
  SlotVoid,
  SlotRefName,

  // Intrinsic function.
  SlotOps,
  SlotIntrinsic,

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
    void *tensor_;
    ops::Op op;
    intrinsic::IntrinsicType intr;
  } value;
};
using Argument = Slot; // Argument is also a Slot.
using Result = Slot;   // Result is also a Slot.

struct Frame {
  CodeType type;
  size_t code;
  size_t pc{0};
  std::vector<Slot> slots;                     // Slot stack.
  std::vector<Slot> vars;                      // Variables by offset.
  std::unordered_map<std::string, Slot> names; // Name map.
};

inline const char *GetSlotTypeStr(const Slot &slot) {
  switch (slot.type) {
  case SlotBool: {
    return "bool";
  }
  case SlotInt: {
    return "int";
  }
  case SlotFloat: {
    return "float";
  }
  case SlotString: {
    return "str";
  }
  case SlotFunction: {
    return "function";
  }
  case SlotGraph: {
    return "graph";
  }
  case SlotClass: {
    return "class";
  }
  case SlotVoid: {
    return "void";
  }
  case SlotRefName: {
    return "ref";
  }
  case SlotTensor: {
    return "tensor";
  }
  case SlotOps: {
    return "ops";
  }
  case SlotIntrinsic: {
    return "intrinsic";
  }
  case SlotInvalid: {
    return "<invalid>";
  }
  default:
    // unknown
    return "<unknown>";
  }
}

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
  case SlotGraph: {
    ss << "graph:" << slot.value.addr;
    break;
  }
  case SlotClass: {
    ss << "class:" << slot.value.addr;
    break;
  }
  case SlotVoid: {
    ss << "void";
    break;
  }
  case SlotRefName: {
    ss << "ref:" << slot.value.addr;
    break;
  }
  case SlotTensor: {
    ss << "tensor:" << slot.value.tensor_;
    break;
  }
  case SlotOps: {
    ss << "ops:" << ops::ToStr(slot.value.op);
    break;
  }
  case SlotIntrinsic: {
    ss << "intrinsic:" << slot.value.addr;
    break;
  }
  case SlotInvalid: {
    ss << "<invalid>";
    break;
  }
  default:
    // unknown
    ss << "<unknown>(" << slot.type << ")";
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
  VM(Compiler *compiler, bool singleFunctionMode = false)
      : codes_{compiler->codes()}, filename_{compiler->filename()},
        singleFunctionMode_{singleFunctionMode} {
    InitInstructionHandlers();
  }
  virtual ~VM() = default;

  Result Run(const std::vector<Argument> &args = std::vector<Argument>());

  runtime::GraphExecutor &graphExecutor() { return graphExecutor_; }

private:
  void InstLoadConst(ssize_t offset);
  void InstLoadName(ssize_t offset);
  void InstStoreName(ssize_t offset);
  void InstLoadLocal(ssize_t offset);
  void InstStoreLocal(ssize_t offset);
  void InstLoadGlobal(ssize_t offset);
  void InstStoreGlobal(ssize_t offset);
  void InstLoadIntrin(ssize_t offset);
  void InstLoadOps(ssize_t offset);
  void InstPopTop(ssize_t offset);
  void InstBinaryAdd(ssize_t offset);
  void InstBinarySub(ssize_t offset);
  void InstBinaryMul(ssize_t offset);
  void InstBinaryDiv(ssize_t offset);
  void InstCompare(ssize_t offset);
  void InstDoCall(ssize_t offset);
  void InstCallIntrin(ssize_t offset);
  void InstCallOps(ssize_t offset);
  void InstReturnVal(ssize_t offset);
  void InstDefineFunc(ssize_t offset);
  void InstDefineGraph(ssize_t offset);
  void InstEnterBlock(ssize_t offset);
  void InstJumpTrue(ssize_t offset);
  void InstJumpFalse(ssize_t offset);
  void InstJump(ssize_t offset);
  void InstStdCin(ssize_t offset);
  void InstStdCout(ssize_t offset);

  void PrepareArguments(Frame &topFrame, const std::vector<Argument> &args);

  std::vector<Slot> &CurrentStack();
  std::vector<Slot> &LocalVars();
  std::vector<Slot> &GlobalVars();

  std::unordered_map<std::string, Slot> &names();

  StringPool &stringPool();

  const std::vector<Code> &codes() const;
  const Code &code() const;

  const std::vector<std::string> &LocalSyms() const;
  const std::vector<std::string> &GlobalSyms() const;

  const std::vector<Constant> &consts() const;
  const std::vector<InstCall> &insts() const;

  Slot *FindLoadedName(const std::string &str);
  bool SetLoadedName(const std::string &str, Slot &&slot);

  void InitInstructionHandlers();

  const std::string &filename() const;
  std::string LineString();

  Slot ConvertConstType(ConstType type, const std::string &value);
  Slot ConvertConstType(const Constant &cons);

  bool ReplaceEscapeStr(std::string &dst);

  bool SkipFuncDefine(const InstCall &inst, size_t &funcDefDepth);

  void DumpStack();

  bool StartGraph(const Code &code);
  void FinishGraph(const Frame &frame);
  void AddGraphParameter(const Code &code, const Slot &arg);

  const std::vector<Code> codes_;

  std::string filename_;

  StringPool stringPool_;

  std::vector<Frame> frames_; // Block, function or module stack.

  Frame *frame_;

  InstHandlerFunctions instHandlers_; // Notice: Do not change.

  runtime::GraphExecutor graphExecutor_;

  bool singleFunctionMode_{false};
};

#define BINARY_OP(OpName, OpSymbol)                                            \
  void VM::InstBinary##OpName(ssize_t offset) {                                \
    const auto &rhs = std::move(CurrentStack().back());                        \
    CurrentStack().pop_back();                                                 \
    const auto &lhs = std::move(CurrentStack().back());                        \
    CurrentStack().pop_back();                                                 \
    if (lhs.type == SlotInt && rhs.type == SlotInt) {                          \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.int_;                                            \
      if (rhsVal == 0 && strcmp(#OpSymbol, "/") == 0) {                        \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
      auto res = lhsVal OpSymbol rhsVal;                                       \
      Slot slot;                                                               \
      slot.type = SlotInt;                                                     \
      slot.value.int_ = res;                                                   \
      LOG_OUT << "result: " << ToString(lhs) << ' ' << #OpSymbol << ' '        \
              << ToString(rhs) << " = " << ToString(slot);                     \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotFloat && rhs.type == SlotFloat) {               \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.float_;                                          \
      if (rhsVal == 0.0 && strcmp(#OpSymbol, "/") == 0) {                      \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
      auto res = lhsVal OpSymbol rhsVal;                                       \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      LOG_OUT << "result: " << ToString(lhs) << ' ' << #OpSymbol << ' '        \
              << ToString(rhs) << " = " << ToString(slot);                     \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotInt && rhs.type == SlotFloat) {                 \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.float_;                                          \
      if (rhsVal == 0.0 && strcmp(#OpSymbol, "/") == 0) {                      \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
      auto res = ((double)lhsVal)OpSymbol rhsVal;                              \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      LOG_OUT << "result: " << ToString(lhs) << ' ' << #OpSymbol << ' '        \
              << ToString(rhs) << " = " << ToString(slot);                     \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotFloat && rhs.type == SlotInt) {                 \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.int_;                                            \
      if (rhsVal == 0 && strcmp(#OpSymbol, "/") == 0) {                        \
        CompileMessage(LineString(), "error: should not div 0");               \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
      auto res = lhsVal OpSymbol((double)rhsVal);                              \
      Slot slot;                                                               \
      slot.type = SlotFloat;                                                   \
      slot.value.float_ = res;                                                 \
      LOG_OUT << "result: " << ToString(lhs) << ' ' << #OpSymbol << ' '        \
              << ToString(rhs) << " = " << ToString(slot);                     \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else if (lhs.type == SlotString || rhs.type == SlotString) {             \
      if (strcmp(#OpSymbol, "+") != 0) {                                       \
        CompileMessage(LineString(),                                           \
                       "error: only support '+' for string operation.");       \
        exit(EXIT_FAILURE);                                                    \
      }                                                                        \
      std::stringstream ss;                                                    \
      GetSlotStr(lhs, ss);                                                     \
      GetSlotStr(rhs, ss);                                                     \
      Slot slot;                                                               \
      slot.type = SlotString;                                                  \
      const char *str = stringPool().Intern(std::move(ss.str()));              \
      slot.value.str_ = str;                                                   \
      LOG_OUT << "result: " << ToString(lhs) << ' ' << #OpSymbol << ' '        \
              << ToString(rhs) << " = " << ToString(slot);                     \
      CurrentStack().emplace_back(std::move(slot));                            \
    } else {                                                                   \
      CompileMessage(LineString(),                                             \
                     "error: only support int, float or string "               \
                     "binary operation[" #OpSymbol "], but got {" +            \
                         ToString(lhs) + ", " + ToString(rhs) + "}.");         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

#define COMPARE_OP(CompareOpSymbol)                                            \
  bool operator CompareOpSymbol(const Slot &lhs, const Slot &rhs) {            \
    if (lhs.type == SlotInt && rhs.type == SlotInt) {                          \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.int_;                                            \
      auto res = lhsVal CompareOpSymbol rhsVal;                                \
      LOG_OUT << "compare II result: " << (res ? "true" : "false") << ", "     \
              << ToString(lhs) << ' ' << #CompareOpSymbol << ' '               \
              << ToString(rhs);                                                \
      return res;                                                              \
    } else if (lhs.type == SlotFloat && rhs.type == SlotFloat) {               \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.float_;                                          \
      auto res = lhsVal CompareOpSymbol rhsVal;                                \
      LOG_OUT << "compare FF result: " << (res ? "true" : "false") << ", "     \
              << ToString(lhs) << ' ' << #CompareOpSymbol << ' '               \
              << ToString(rhs);                                                \
      return res;                                                              \
    } else if (lhs.type == SlotInt && rhs.type == SlotFloat) {                 \
      auto lhsVal = lhs.value.int_;                                            \
      auto rhsVal = rhs.value.float_;                                          \
      auto res = ((double)lhsVal)CompareOpSymbol rhsVal;                       \
      LOG_OUT << "compare IF result: " << (res ? "true" : "false") << ", "     \
              << ToString(lhs) << ' ' << #CompareOpSymbol << ' '               \
              << ToString(rhs);                                                \
      return res;                                                              \
    } else if (lhs.type == SlotFloat && rhs.type == SlotInt) {                 \
      auto lhsVal = lhs.value.float_;                                          \
      auto rhsVal = rhs.value.int_;                                            \
      auto res = lhsVal CompareOpSymbol((double)rhsVal);                       \
      LOG_OUT << "compare FI result: " << (res ? "true" : "false") << ", "     \
              << ToString(lhs) << ' ' << #CompareOpSymbol << ' '               \
              << ToString(rhs);                                                \
      return res;                                                              \
    } else if (lhs.type == SlotString && rhs.type == SlotString) {             \
      auto res = strcmp(lhs.value.str_, rhs.value.str_) CompareOpSymbol 0;     \
      LOG_OUT << "compare STR result: " << (res ? "true" : "false") << ", "    \
              << ToString(lhs) << ' ' << #CompareOpSymbol << ' '               \
              << ToString(rhs);                                                \
      return res;                                                              \
    } else {                                                                   \
      throw std::runtime_error("Unexcepted constant");                         \
    }                                                                          \
  }
} // namespace vm
} // namespace da
#endif // __VM_VM_H__