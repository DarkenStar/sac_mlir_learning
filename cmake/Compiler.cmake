# Include a custom CMake module for logging utilities from the project's cmake directory
include(${CMAKE_SOURCE_DIR}/cmake/Logging.cmake)

# Enable exporting of all build commands for debugging purposes
# This generates a log of commands in files like CMakeFiles/CMakeOutput.log
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the C++ standard to C++17 for the project
set(CMAKE_CXX_STANDARD 17)

# Enforce C++17 as a strict requirement
# CMake will fail if the compiler does not support C++17
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check if the compiler is Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # Disable Run-Time Type Information (RTTI) to reduce binary size and improve performance
    # This adds -fno-rtti to the compiler flags, disabling features like dynamic_cast and typeid
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

    # Add -O3 optimization flag for release builds
    # -O3 enables aggressive optimizations for better performance
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

    # Add -g flag for debug builds to include debugging symbols
    # This allows tools like gdb to provide detailed debugging information
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
else()
    # If the compiler is not Clang, log a fatal error and stop configuration
    # log_fatal is defined in Logging.cmake
    log_fatal("Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
endif()