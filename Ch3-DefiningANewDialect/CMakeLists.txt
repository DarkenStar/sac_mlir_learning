project(Ch3-DefiningANewDialect)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

set(PROJECT_TARGET_PREFIX "ch3")
set(CMAKE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/${PROJECT_NAME})
mlir_learn_log_info("|- Output Directory: ${CMAKE_OUTPUT_DIR}")

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_OUTPUT_DIR}/include
)

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")