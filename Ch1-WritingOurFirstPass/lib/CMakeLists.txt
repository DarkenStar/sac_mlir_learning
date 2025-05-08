set(TARGET_NAME "ch1-tutorial-lib")
mlir_learn_log_info("|- Target <lib>: ${TARGET_NAME}")

add_mlir_library(

    # Library Name:
    ${TARGET_NAME}

    # Source Files:
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform/Affine/AffineFullUnroll.cpp

    # Link Libraries:
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRAffineDialect
    MLIRFuncDialect
    MLIRInferTypeOpInterface
)

# target_include_directories(<target> [SYSTEM] [BEFORE | AFTER]
# <INTERFACE | PUBLIC | PRIVATE> <directory> [<directory>...])
# PRIVATE: 仅对当前目标有效，头文件路径不会传递给依赖当前目标的其他目标。
# PUBLIC: 对当前目标有效，并且会传递给依赖当前目标的其他目标。
# INTERFACE: 仅对依赖当前目标的其他目标有效，当前目标本身不使用这些路径。
target_include_directories(
    ${TARGET_NAME}
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
# set_target_properties(<target1> [<target2> ...]# PROPERTIES <property1> <value1> [<property2> <value2> ...])
# RUNTIME_OUTPUT_DIRECTORY: 设置可执行文件的输出目录
# LIBRARY_OUTPUT_DIRECTORY: 设置动态库的输出目录
# ARCHIVE_OUTPUT_DIRECTORY: 设置静态库的输出目录
set_target_properties(${TARGET_NAME} PROPERTIES

    # Output Directory:
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_OUTPUT_DIR}/lib
)