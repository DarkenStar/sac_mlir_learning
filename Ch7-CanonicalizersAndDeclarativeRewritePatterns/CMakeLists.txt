project(Ch7-CanonicalizersAndDeclarativeRewritePatterns)
mlir_learn_log_info("${MLIR_LEARN_LOG_PURPLE}Project: ${PROJECT_NAME}${MLIR_LEARN_LOG_RESET}")

set(PROJECT_TARGET_PREFIX "ch7")

add_subdirectory("include")
add_subdirectory("lib")
add_subdirectory("tools")