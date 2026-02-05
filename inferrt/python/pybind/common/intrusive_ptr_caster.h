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

#ifndef __COMMON_INTRUSIVE_PTR_CASTER_H__
#define __COMMON_INTRUSIVE_PTR_CASTER_H__

#include <nanobind/nanobind.h>
#include "ir/common/intrusive_ptr.h"

namespace nanobind {
namespace detail {

template <typename T>
struct type_caster<mrt::ir::IntrusivePtr<T>> {
  using Holder = mrt::ir::IntrusivePtr<T>;
  NB_TYPE_CASTER(Holder, make_caster<T>::Name)

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    make_caster<T> caster;
    if (!caster.from_python(src, flags, cleanup)) return false;
    value = Holder(caster.operator T *());
    return true;
  }

  static handle from_cpp(const Holder &ptr, rv_policy policy, cleanup_list *cleanup) noexcept {
    if (!ptr) return none().release();

    // Return a reference to the raw pointer.
    handle h = make_caster<T>::from_cpp(ptr.get(), rv_policy::reference, cleanup);

    if (h.ptr()) {
      // Keep the intrusive pointer alive as long as the Python object exists.
      // We use a capsule to hold a copy of the IntrusivePtr.
      Holder *holder_copy = new Holder(ptr);

      handle cap = capsule(holder_copy, [](void *p) noexcept { delete static_cast<Holder *>(p); }).release();

      // Attach the capsule to the Python object.
      // This ensures the IntrusivePtr stays alive (and hence the RefCounted object)
      // until the Python object is destroyed.
      PyObject_SetAttrString(h.ptr(), "__mrt_holder__", cap.ptr());
      Py_DECREF(cap.ptr());
    }

    return h;
  }
};

}  // namespace detail
}  // namespace nanobind

#endif
