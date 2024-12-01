#include "ir/namespace.h"

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