set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-tutorial/Transform/Affine/Passes.td)
mlir_tablegen(mlir-tutorial/Transform/Affine/Passes.h.inc -gen-pass-decls -name=Affine)
mlir_tablegen(mlir-tutorial/Transform/Affine/AffinePasses.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(
    ALL_TABLEGEN_TARGETS
    ${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen
)
mlir_learn_log_info("|- Target <tablegen>: ${ALL_TABLEGEN_TARGETS}")

set_property(
    GLOBAL PROPERTY ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
    ${ALL_TABLEGEN_TARGETS}
)