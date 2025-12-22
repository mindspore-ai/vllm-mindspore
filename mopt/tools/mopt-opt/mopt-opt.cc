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

// MLIR infrastructure includes
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"

// MLIR core components
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"

// LLVM utilities and support
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"

// External dialect registrations
#include "stablehlo/dialect/Register.h"
#include "torch-mlir/InitAll.h"

// Mopt project specific includes
#include "mopt/Dialect/Mrt/Mrt.h"
#include "mopt/Dialect/Dvm/DvmDialect.h"
#include "mopt/Conversion/Passes.h"
#include "mopt/Fusion/Passes.h"
#include "mopt/Dialect/Mrt/Transforms/Passes.h"

namespace {
// Version information for mopt-opt tool
constexpr int kMoptOptMajorVersion = 1;
constexpr int kMoptOptMinorVersion = 0;

// Helper function to initialize the registry
void initializeDialectRegistry(mlir::DialectRegistry &registry) {
  mlir::registerAllDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  mlir::torch::registerAllDialects(registry);
  mlir::torch::registerAllExtensions(registry);
  registry.insert<mrt::MrtDialect>();
  registry.insert<mlir::dvm::DvmDialect>();
}
}  // namespace

int main(int argc, char **argv) {
  // Initialize all standard MLIR passes
  mlir::registerAllPasses();

  // Register Torch-MLIR passes
  mlir::torch::registerAllPasses();

  // Register custom conversion passes for Mopt
  mlir::registerMoptConversionPasses();

  // Register Mopt fusion passes (StableHLO outlining, symbolic shape propagation, etc.)
  mlir::registerMoptFusionPasses();

  // Register MRT transforms passes
  mlir::registerMrtTransformsPasses();

  // Setup dialect registry with all required dialects
  mlir::DialectRegistry registry;
  initializeDialectRegistry(registry);

  return mlir::asMainReturnCode(mlir::MlirOptMain(argc, argv, "Mopt optimizer driver\n", registry));
}
