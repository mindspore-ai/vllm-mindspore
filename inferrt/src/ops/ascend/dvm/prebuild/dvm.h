/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#ifndef _DVM_H_
#define _DVM_H_

#include <cstdint>
#include <vector>

namespace dvm {
enum DType {
  kBool = 0,
  kFloat16,
  kBFloat16,
  kFloat32,
  kInt32,
  kInt64,
  kTypeEnd,
};

enum UnaryOpType {
  kSqrt = 0,
  kAbs,
  kLog,
  kExp,
  kReciprocal,
  kIsFinite,
  kLogicalNot,
  kRound,
  kFloor,
  kCeil,
  kTrunc,
  kUnaryOpEnd,
};

enum GroupType {
  kSplit_M = 0,
  kSplit_N,
  kSplit_K,
  kGroupTypeEnd,
};

enum BinaryOpType {
  kEqual = 0,
  kNotEqual,
  kGreater,
  kGreaterEqual,
  kLess,
  kLessEqual,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kPow,
  kMaximum,
  kMinimum,
  kLogicalAnd,
  kLogicalOr,
  kBinaryOpEnd,
};

enum ReduceOpType {
  kSum = 0,
  kReduceOpEnd,
};

enum KernelType {
  kStaticShape = 0,
  kDynShape,
  kStaticParallel,
  kStaticMix,
  kDynMix,
  kStaticStages,
  kEager,
  kKernelTypelEnd,
};

class NDObject;
class VKernel;
class MsProfHelper;
class Communicator;

struct ShapeRef {
  ShapeRef() {}
  explicit ShapeRef(const std::vector<int64_t> &other) : data(other.data()), size(other.size()) {}
  ShapeRef &operator=(const std::vector<int64_t> &other) {
    data = other.data();
    size = other.size();
    return *this;
  }
  const int64_t *data;
  size_t size;
};

class Float16 {
 public:
  explicit Float16(const uint16_t &v) : value_(v) {}
  explicit Float16(const float &v);
  explicit Float16(const int32_t &v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

class BFloat16 {
 public:
  explicit BFloat16(const uint16_t &v) : value_(v) {}
  explicit BFloat16(const float &v);
  explicit BFloat16(const int32_t &v);
  explicit operator float() const;
  explicit operator int32_t() const;
  uint16_t int_value() const { return value_; }

 private:
  uint16_t value_;
};

struct RelocTable {
  NDObject **inputs;
  size_t inputs_size;
  NDObject **outputs;
  size_t outputs_size;
};

struct RelocEntry {
  RelocEntry() {}
  RelocEntry(NDObject *p, void *a) : io(p), addr(a) {}
  NDObject *io;
  void *addr;
};

typedef void *(*WsAllocFunc)(uint64_t size, void *user_data);

class Comm {
 public:
  Comm() = default;
  ~Comm();
  bool Init(int rank_id, int rank_size);
  inline const Communicator *GetImpl() const { return comm_; }

 private:
  Communicator *comm_{nullptr};
};

class Kernel {
 public:
  Kernel();
  ~Kernel();

  void Reset(KernelType type);
  int ParallelNext();

  NDObject *Load(void *addr, ShapeRef *shape, DType type);
  NDObject *SliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *size, DType type);
  NDObject *StridedSliceLoad(void *addr, ShapeRef *shape, ShapeRef *start, ShapeRef *end, ShapeRef *step, DType type);
  NDObject *MultiLoad(void *addr, ShapeRef *shape, DType type, const Comm *comm);
  NDObject *Store(void *addr, NDObject *input);
  NDObject *PadStore(void *addr, NDObject *input, int64_t pad_size);
  void SetStoreInplace(NDObject *store);

  NDObject *Unary(int op_type, NDObject *input);
  NDObject *Binary(int op_type, NDObject *lhs, NDObject *rhs);
  template <typename T>
  NDObject *Binary(int op_type, T val, NDObject *rhs);
  template <typename T>
  NDObject *Binary(int op_type, NDObject *lhs, T val);

  NDObject *Reduce(int op_type, NDObject *input, ShapeRef *dims, bool keepdims);
  NDObject *Select(NDObject *cond, NDObject *lhs, NDObject *rhs);

  NDObject *Cast(NDObject *input, DType type);
  NDObject *Broadcast(NDObject *input, ShapeRef *shape);

  template <typename T>
  NDObject *Broadcast(T val, ShapeRef *shape, DType type, bool dummy_load);
  NDObject *Reshape(NDObject *input, ShapeRef *shape);
  NDObject *Copy(NDObject *input);

  template <typename T>
  NDObject *OneHot(NDObject *indices, ShapeRef *depth, int axis, T on_value, T off_value);

  NDObject *ElemAny(NDObject *input);

  NDObject *MatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias);
  NDObject *GroupedMatMul(NDObject *lhs, NDObject *rhs, bool trans_a, bool trans_b, NDObject *bias,
                          NDObject *group_list, GroupType group_type);

  // collective communication
  NDObject *AllReduce(NDObject *input, const Comm *comm);
  NDObject *AllGather(NDObject *input, const Comm *comm);
  NDObject *AllGatherV2(NDObject *input, const Comm *comm);
  NDObject *ReduceScatter(NDObject *input, const Comm *comm);

  void StageSwitch(KernelType type);
  NDObject *StageLoad(NDObject *stage_store);
  NDObject *StageStore(NDObject *input);
  NDObject *StagePadStore(NDObject *input, int64_t pad_size);

  uint64_t CodeGen();
  // only used for dynamic-shape kernel shape inference
  void Infer();
  int Launch(void *workspace, void *stream);
  int Launch(const RelocTable &reloc_table, void **inputs, void **outputs, void *workspace, void *stream);
  int MsProfLaunch(const char *op_name, const char *op_fullname, const RelocTable &reloc_table, void **inputs,
                   void **outputs, void *workspace, void *stream);

  void EagerReset(WsAllocFunc ws_alloc, void *user_data);
  void EagerCodeGen(const RelocEntry *reloc_table, size_t reloc_size);
  int EagerLaunch(void *stream);
  int EagerMsProfLaunch(void *stream);
  void EagerClear();

  ShapeRef *GetShape(NDObject *op) const;
  DType GetDType(NDObject *op) const;

  const char *Dump() const;
  const char *Das() const;

  VKernel *GetImpl() const { return kernel_; }

 private:
  VKernel *kernel_;
  MsProfHelper *msprof_helper_;
};

void SetDeterministic(bool enable);

void SetOnlineTuning(bool enable);

}  // namespace dvm
#endif  // _DVM_H_
