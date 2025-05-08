cmake_minimum_required(VERSION 3.30)

project("MLIR-LEARN")

include(${CMAKE_SOURCE_DIR}/cmake/Logging.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Compiler.cmake)

# find_package(<PackageName> [version] [EXACT] [QUIET] [REQUIRED] [COMPONENTS component1 component2 ...])
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

mlir_learn_log_info("Using MLIRConfig.cmake in: ${MLIR_DIR}")
mlir_learn_log_info("Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MILR_BINARY_DIR ${CMAKE_BINARY_DIR})

# list(<operation> <list_variable> [args...])
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# link_directories([BEFORE | AFTER] <directory1> [<directory2> ...])
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS}) # macro definitions

add_subdirectory("Ch1-WritingOurFirstPass")
add_subdirectory("Ch2-UsingTableGenForPasses")
add_subdirectory("Ch3-DefiningANewDialect")
add_subdirectory("Ch4-UsingTraits")
add_subdirectory("Ch5-FoldersAndConstantPropagation")
add_subdirectory("Ch6-Verifiers")
add_subdirectory("Ch7-CanonicalizersAndDeclarativeRewritePatterns")
add_subdirectory("Ch8-DialectConversion")
add_subdirectory("Ch9-LowerPolyToLLVM")