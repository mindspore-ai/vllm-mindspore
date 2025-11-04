add_mlir_dialect(Mrt mrt)
add_mlir_doc(Mrt Mrt Dialect/ -gen-op-doc)

# Additional TableGen targets for attributes and types
set(LLVM_TARGET_DEFINITIONS Mrt.td)
mlir_tablegen(MrtAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=mrt)
mlir_tablegen(MrtAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=mrt)
mlir_tablegen(MrtTypes.h.inc -gen-typedef-decls -typedefs-dialect=mrt)
mlir_tablegen(MrtTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=mrt)
mlir_tablegen(MrtEnums.h.inc -gen-enum-decls)
mlir_tablegen(MrtEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRMrtExtraIncGen)