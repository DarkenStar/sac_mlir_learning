set(TARGET_NAME ${PROJECT_TARGET_PREFIX}-tutorial-opt)
mlir_learn_log_info("|- Target <exe>: ${TARGET_NAME}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(
    TOOL_LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    ${PROJECT_TARGET_PREFIX}-tutorial-lib
)

add_llvm_executable(${TARGET_NAME} tutorial-opt.cpp)

set_target_properties(
    ${TARGET_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/tools
)

llvm_update_compile_flags(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE ${TOOL_LIBS})

mlir_check_all_link_libraries(${TARGET_NAME})