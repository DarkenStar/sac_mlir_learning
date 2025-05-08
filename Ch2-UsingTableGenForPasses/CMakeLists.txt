project(Ch2-UsingTableGenForPasses)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

set(PROJECT_TARGET_PREFIX "ch2")
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)