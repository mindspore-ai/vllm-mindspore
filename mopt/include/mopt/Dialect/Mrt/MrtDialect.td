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

#ifndef MRT_DIALECT_MRT_DIALECT_TD
#define MRT_DIALECT_MRT_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Mrt_Dialect : Dialect {
  let name = "mrt";
  let cppNamespace = "::mrt";

  let description = [{
    The MRT (Machine Learning Runtime) dialect provides a set of operations and
    types for representing machine learning computations. This dialect is designed
    to be a high-level MLIR dialect that can be lowered to various targets.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Mrt_Type<string name, string typeMnemonic> : TypeDef<Mrt_Dialect, name> {
  let mnemonic = typeMnemonic;
}

// Base class for all MRT dialect operations.
class Mrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mrt_Dialect, mnemonic, traits> {
  // Whether generate aclnn launch code.
  // When set to '1', the aclnn launch code will be auto generated.
  bit generated = 0;
}

#endif // MRT_DIALECT_MRT_DIALECT_TD
