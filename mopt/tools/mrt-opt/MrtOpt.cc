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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/IR/AsmState.h"

#include "mopt/Dialect/Mrt/Mrt.h"

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register any command line options.
  mlir::registerMLIRContextCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerPassManagerCLOptions();

  // Register passes here.

  mlir::DialectRegistry registry;
  registry.insert<mrt::MrtDialect, mlir::func::FuncDialect>();

  // registerAllDialects(registry);
  // registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "MLIR MRT optimization driver\n", registry));
}
