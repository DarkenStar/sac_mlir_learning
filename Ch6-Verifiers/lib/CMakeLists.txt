set(TARGET_NAME ${PROJECT_TARGET_PREFIX}-tutorial-lib)
mlir_learn_log_info("|- Target <lib>: ${TARGET_NAME}")

# Get the tablegen targets from the global property;
# Check "../include/CMakeLists.txt" for its definition.
get_property(
    TABLEGEN_TARGETS GLOBAL PROPERTY
    ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
)

# If you want to append your library to ${MLIR_DIALECT_LIBS},
# Use:
# > add_mlir_dialect_library(...)
# Otherwise, use:
# > add_mlir_library(...)
add_mlir_library(

    # Library Name:
    ${TARGET_NAME}

    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Poly/PolyDialect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Dialect/Poly/PolyOps.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnrollPatternRewrite.cpp

    DEPENDS
    ${TABLEGEN_TARGETS}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
)

target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/include
)

set_target_properties(${TARGET_NAME} PROPERTIES

    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/lib
)