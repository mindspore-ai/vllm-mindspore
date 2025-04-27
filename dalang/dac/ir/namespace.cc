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

#include "ir/namespace.h"

namespace da {
namespace ir {
// Namespace pool.
static Namespaces gNsPool;
NamespacePtr NewNamespace() {
  (void)gNsPool.emplace_back(new Namespace());
  return gNsPool.back();
}
void ClearNamespacePool() {
  for (NamespacePtr ns : gNsPool) {
    delete ns;
  }
  gNsPool.clear();
}

// Block pool.
static Blocks gBlockPool;
BlockPtr NewBlock() {
  (void)gBlockPool.emplace_back(new Block());
  return gBlockPool.back();
}
void ClearBlockPool() {
  for (BlockPtr block : gBlockPool) {
    delete block;
  }
  gBlockPool.clear();
}

// Func pool.
static Funcs gFuncPool;
FuncPtr NewFunc() {
  (void)gFuncPool.emplace_back(new Func());
  return gFuncPool.back();
}
void ClearFuncPool() {
  for (FuncPtr func : gFuncPool) {
    delete func;
  }
  gFuncPool.clear();
}
} // namespace ir
} // namespace da