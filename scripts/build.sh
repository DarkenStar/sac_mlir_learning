export LLVM_PROJECT_DIR="/code/llvm-project"
export LLVM_BUILD_DIR="${LLVM_PROJECT_DIR}/build" # Define build dir variable

# Clean previous build artifacts
#rm -rf build

# Set compilers explicitly for CMake
export CC=clang
export CXX=clang++

# Configure CMake
# Explicitly point to the LLVM/MLIR build directory's CMake config location
cmake -S . -G Ninja -B build \
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=${CC} \
    -DCMAKE_CXX_COMPILER=${CXX}

# Build the project
cmake --build build -j 8