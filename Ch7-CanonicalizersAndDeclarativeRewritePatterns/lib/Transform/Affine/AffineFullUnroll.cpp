#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopFusionUtils.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir-tutorial/Transform/Affine/AffineFullUnroll.hpp"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tutorial {
#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "mlir-tutorial/Transform/Affine/Passes.h.inc"

using ::mlir::affine::AffineForOp;
using ::mlir::affine::loopUnrollFull;

// A pass that manually walks the IR
struct AffineFullUnroll: impl::AffineFullUnrollBase<AffineFullUnroll>
{
    using AffineFullUnrollBase::AffineFullUnrollBase;

    void runOnOperation() override
    {
        getOperation()->walk([&](AffineForOp op) {
            if (failed(loopUnrollFull(op))) {
                op->emitError("Unrolling Failed");
                signalPassFailure();
            }
        });
    }
};
}