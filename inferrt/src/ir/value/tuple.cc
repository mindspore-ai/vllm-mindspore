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

#include "ir/value/tuple.h"
#include "ir/value/value.h"

namespace mrt {
namespace ir {

Tuple::Tuple() : Tuple(std::vector<Value>{}) {}

Tuple::Tuple(const std::vector<Value> &elements) : impl_(MakeIntrusive<TupleImpl>(elements)) {}

Tuple::Tuple(const std::vector<Value> &&elements) : impl_(MakeIntrusive<TupleImpl>(std::move(elements))) {}

Tuple::~Tuple() = default;

Tuple::Tuple(const Tuple &other) = default;

Tuple::Tuple(Tuple &&other) noexcept = default;

Tuple &Tuple::operator=(const Tuple &other) = default;

Tuple &Tuple::operator=(Tuple &&other) noexcept = default;

const std::vector<Value> &Tuple::GetElements() const { return impl_->elements; }

bool Tuple::Defined() const { return bool(impl_); }

}  // namespace ir
}  // namespace mrt
