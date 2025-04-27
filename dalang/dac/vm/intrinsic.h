/**
 * Copyright 2025 Zhang Qinghua
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

#ifndef __VM_INTRINSIC_H__
#define __VM_INTRINSIC_H__

#include "tensor/tensor.h"

namespace da {
namespace intrinsic {

#define TYPE(t) IntrinsicType_##t,
enum IntrinsicType {
#include "lexer/literal_type.list"
  IntrinsicType_print,
};
#undef TYPE

} // namespace intrinsic
} // namespace da

#endif // __VM_INTRINSIC_H__