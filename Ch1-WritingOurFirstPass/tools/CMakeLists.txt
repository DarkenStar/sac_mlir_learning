set(TARGET_NAME "ch1-tutorial-opt")
mlir_learn_log_info("|- Target <exe>: ${TARGET_NAME}")

# get_property(<variable> <scope> PROPERTY <property_name>
# [SET | DEFINED | BRIEF_DOCS | FULL_DOCS])  
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIRArithDialect
    MLIROptLib
    ch1-tutorial-lib)

add_llvm_executable(${TARGET_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/tutorial-opt.cpp)

set_target_properties(${TARGET_NAME} PROPERTIES
    # Ouput Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/tools)

# target_link_libraries(<target>
# [PRIVATE | PUBLIC | INTERFACE]
# <item>...
# [<PRIVATE | PUBLIC | INTERFACE> <item>...]...)

llvm_update_compile_flags(${TARGET_NAME})
target_link_libraries(${TARGET_NAME} PRIVATE ${LIBS})


mlir_check_all_link_libraries(${TARGET_NAME})