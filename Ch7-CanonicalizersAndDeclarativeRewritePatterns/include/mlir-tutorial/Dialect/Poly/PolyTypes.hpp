#pragma once

// Required because the .h.inc file refers to MLIR classes and does not itself
// have any includes.
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp.inc"

