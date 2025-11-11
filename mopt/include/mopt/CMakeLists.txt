add_subdirectory(Dialect/Mrt)

# Generate pass declarations
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Mopt)
add_public_tablegen_target(MoptPassIncGen)