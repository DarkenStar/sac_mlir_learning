#pragma once

#include <mlir/Pass/Pass.h>

namespace mlir::tutorial
{

#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "mlir-tutorial/Transform/Affine/Passes.h.inc"
}  // namespace mlir::tutorial