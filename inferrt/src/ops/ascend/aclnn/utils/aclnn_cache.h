/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__
#define __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__

#include <cstddef>
#include <vector>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include <utility>

#include "common/common.h"
#include "ir/value/value.h"
#include "ir/common/intrusive_ptr.h"
#include "ops/ascend/aclnn/utils/aclnn_common_meta.h"
#include "ops/ascend/aclnn/utils/aclnn_deleter.h"
#include "ops/ascend/aclnn/utils/convert_utils.h"

namespace mrt {
namespace ops {
// cache process type
enum class CacheReleaseType {
  RELEASE_PARAMS,               // release converted params
  RELEASE_EXECUTOR,             // release executor
  RELEASE_PARAMS_AND_EXECUTOR,  // release converted params and executor
};

// Base class for aclnn cache with reference counting
class CacheEntry : public ir::RefCounted {
 public:
  CacheEntry() = default;
  virtual ~CacheEntry() = default;

  virtual void Release(const CacheReleaseType &type) = 0;
  virtual void UpdateTensorAddr(size_t *index, size_t *relativeIndex, void *tensorAddr) = 0;
  virtual aclOpExecutor *GetExecutor() = 0;

 private:
  DISABLE_COPY_AND_ASSIGN(CacheEntry)
};
using CacheEntryPtr = ir::IntrusivePtr<CacheEntry>;

// Update tensor address
template <typename T>
inline void UpdateAddr(const CacheEntryPtr &cacheEntry, const T &value, size_t *index) {
  LOG_OUT << "UpdateAddr for non tensor type, index: " << *index;
  ++(*index);
}

inline void UpdateAddr(const CacheEntryPtr &cacheEntry, const ir::TensorPtr &tensor, size_t *index) {
  cacheEntry->UpdateTensorAddr(index, nullptr, tensor->DataPtr());
  ++(*index);
}

inline void UpdateAddr(const CacheEntryPtr &cacheEntry, const std::vector<ir::TensorPtr> &tensorList, size_t *index) {
  for (size_t i = 0; i < tensorList.size(); ++i) {
    cacheEntry->UpdateTensorAddr(index, &i, tensorList[i]->DataPtr());
  }
  ++(*index);
}

inline void UpdateAddr(const CacheEntryPtr &cacheEntry, const ir::TuplePtr &tuple, size_t *index) {
  if (tuple == nullptr || tuple->Size() == 0) {
    LOG_OUT << "tuple is empty";
    ++(*index);
    return;
  }
  if ((*tuple)[kIndex0]->IsTensor()) {
    std::vector<ir::TensorPtr> tensorList;
    TupleToTensorList(*tuple, &tensorList);
    UpdateAddr(cacheEntry, tensorList, index);
    return;
  }
  ++(*index);
}

// Main entry for update tensor address
template <typename... Args>
void CallUpdateAddr(const CacheEntryPtr &cacheEntry, const Args &...args) {
  size_t index = 0;
  (UpdateAddr(cacheEntry, args, &index), ...);
}

// Update a single tensor address
inline void UpdateAclTensorAddr(aclTensor *tensor, size_t *index, aclOpExecutor *executor, void *tensorAddr) {
  static const auto aclSetTensorAddr = GET_ACLNN_COMMON_META_FUNC(aclSetTensorAddr);
  if (aclSetTensorAddr == nullptr) {
    LOG_EXCEPTION << "aclSetTensorAddr is nullptr";
    return;
  }
  aclSetTensorAddr(executor, *index, tensor, tensorAddr);
}

// Update a tensor list address
inline void UpdateAclTensorListAddr(aclTensorList *tensorList, size_t *index, size_t *relativeIndex,
                                    aclOpExecutor *executor, void *tensorAddr) {
  static const auto aclSetDynamicTensorAddr = GET_ACLNN_COMMON_META_FUNC(aclSetDynamicTensorAddr);
  if (aclSetDynamicTensorAddr == nullptr) {
    LOG_EXCEPTION << "aclSetDynamicTensorAddr is nullptr";
    return;
  }
  aclSetDynamicTensorAddr(executor, *index, *relativeIndex, tensorList, tensorAddr);
}

// Cache processor for cache operations
template <typename Tuple>
class CacheProcessor {
 public:
  explicit CacheProcessor(Tuple &&tuple, aclOpExecutor *executor)
      : convertedParams_(std::move(tuple)), executor_(executor) {
    InitTensorAddrUpdaters();
  }

  CacheProcessor(CacheProcessor &&other) noexcept
      : convertedParams_(std::move(other.convertedParams_)),
        executor_(other.executor_),
        isParamsReleased_(other.isParamsReleased_),
        isExecutorReleased_(other.isExecutorReleased_) {
    other.executor_ = nullptr;
    other.isParamsReleased_ = true;
    other.isExecutorReleased_ = true;
  }

  CacheProcessor &operator=(CacheProcessor &&other) noexcept {
    if (this != &other) {
      if (!isParamsReleased_) {
        ReleaseConvertedParams(convertedParams_);
      }

      convertedParams_ = std::move(other.convertedParams_);
      executor_ = other.executor_;
      isParamsReleased_ = other.isParamsReleased_;
      isExecutorReleased_ = other.isExecutorReleased_;

      other.executor_ = nullptr;
      other.isParamsReleased_ = true;
      other.isExecutorReleased_ = true;
    }
    return *this;
  }

  template <size_t I>
  static void BuildTensorAddrUpdater() {
    using elementType = std::decay_t<std::tuple_element_t<I, Tuple>>;
    if constexpr (std::is_same_v<elementType, aclTensor *>) {
      tensorAddrUpdatersMap_[I] = [](const Tuple &convertedParams, aclOpExecutor *executor, size_t *index,
                                     size_t *relativeIndex, void *tensorAddr) {
        UpdateAclTensorAddr(std::get<I>(convertedParams), index, executor, tensorAddr);
      };
    }
    if constexpr (std::is_same_v<elementType, aclTensorList *>) {
      tensorAddrUpdatersMap_[I] = [](const Tuple &convertedParams, aclOpExecutor *executor, size_t *index,
                                     size_t *relativeIndex, void *tensorAddr) {
        UpdateAclTensorListAddr(std::get<I>(convertedParams), index, relativeIndex, executor, tensorAddr);
      };
    }
  }

  template <size_t... I>
  static void BuildTensorAddrUpdaters(std::index_sequence<I...>) {
    (BuildTensorAddrUpdater<I>(), ...);
  }

  static void InitTensorAddrUpdaters() {
    constexpr size_t tuple_size = std::tuple_size_v<Tuple>;
    static_assert(tuple_size > 0, "Tuple size must be greater than 0");
    static bool isInitialized = false;
    if (isInitialized) {
      return;
    }
    isInitialized = true;
    LOG_OUT << "Initializing tensor address updaters for tuple of size: " << tuple_size;

    BuildTensorAddrUpdaters(std::make_index_sequence<tuple_size>{});
  }

  ~CacheProcessor() {
    // release params and executor
    if (!isParamsReleased_) {
      ReleaseConvertedParams(convertedParams_);
    }
    if (!isExecutorReleased_) {
      ReleaseExecutor(executor_);
    }
  }

  void Release(const CacheReleaseType &type) {
    switch (type) {
      case CacheReleaseType::RELEASE_PARAMS:
        if (!isParamsReleased_) {
          ReleaseConvertedParams(convertedParams_);
          isParamsReleased_ = true;
        }
        break;
      case CacheReleaseType::RELEASE_EXECUTOR:
        if (!isExecutorReleased_) {
          ReleaseExecutor(executor_);
          isExecutorReleased_ = true;
        }
        break;
      case CacheReleaseType::RELEASE_PARAMS_AND_EXECUTOR:
        if (!isParamsReleased_) {
          ReleaseConvertedParams(convertedParams_);
          isParamsReleased_ = true;
        }
        if (!isExecutorReleased_) {
          ReleaseExecutor(executor_);
          isExecutorReleased_ = true;
        }
        break;
      default:
        LOG_EXCEPTION << "Invalid cache release type: " << static_cast<int>(type);
        break;
    }
  }

  void UpdateTensorAddr(size_t *index, size_t *relativeIndex, void *tensorAddr) {
    LOG_OUT << "UpdateTensorAddr called for index: " << *index << ", updaters size: " << tensorAddrUpdatersMap_.size()
            << ", relativeIndex: " << (relativeIndex == nullptr ? 0 : *relativeIndex);

    // Use the static map for efficient lookup, no need lookup in the future
    auto it = tensorAddrUpdatersMap_.find(*index);
    if (it != tensorAddrUpdatersMap_.end()) {
      LOG_OUT << "Found updater for index " << *index;
      it->second(convertedParams_, executor_, index, relativeIndex, tensorAddr);
    } else {
      LOG_EXCEPTION << "No updater found for index: " << *index << ", available indices: ";
      for (const auto &pair : tensorAddrUpdatersMap_) {
        LOG_OUT << pair.first << " ";
      }
    }
  }

  aclOpExecutor *GetExecutor() { return executor_; }

  using TensorAddrUpdater = std::function<void(const Tuple &, aclOpExecutor *, size_t *, size_t *, void *)>;

 private:
  DISABLE_COPY_AND_ASSIGN(CacheProcessor)
  Tuple convertedParams_;
  aclOpExecutor *executor_;

  // Static map for updater functions (no instance data)
  inline static std::unordered_map<size_t, TensorAddrUpdater> tensorAddrUpdatersMap_;

  bool isParamsReleased_{false};
  bool isExecutorReleased_{false};
};

// Wrapper class for CacheEntry
template <typename CacheProcessor>
class CacheEntryImpl : public CacheEntry {
 public:
  explicit CacheEntryImpl(CacheProcessor &&cacheProcessor) : cacheProcessor_(std::move(cacheProcessor)) {}
  ~CacheEntryImpl() override = default;

  void Release(const CacheReleaseType &type) override { cacheProcessor_.Release(type); }

  void UpdateTensorAddr(size_t *index, size_t *relativeIndex, void *tensorAddr) override {
    cacheProcessor_.UpdateTensorAddr(index, relativeIndex, tensorAddr);
  }

  aclOpExecutor *GetExecutor() override { return cacheProcessor_.GetExecutor(); }

 private:
  DISABLE_COPY_AND_ASSIGN(CacheEntryImpl)
  CacheProcessor cacheProcessor_;
};

}  // namespace ops
}  // namespace mrt
#endif  // __OPS_ASCEND_ACLNN_UTILS_ACLNN_CACHE_H__
