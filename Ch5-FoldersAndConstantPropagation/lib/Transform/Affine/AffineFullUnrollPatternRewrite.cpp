#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir-tutorial/Transform/Affine/AffineFullUnrollPatternRewrite.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tutorial
{

#define GEN_PASS_DEF_AFFINEFULLUNROLLPATTERNREWRITE
#include "mlir-tutorial/Transform/Affine/Passes.h.inc"

using ::mlir::affine::AffineForOp;
using ::mlir::affine::loopUnrollFull;

// A pattern that matches on AffineForOp and unrolls it.
struct AffineFullUnrollPattern : public OpRewritePattern<AffineForOp>
{
    AffineFullUnrollPattern(MLIRContext* context)
        : OpRewritePattern<AffineForOp>(context, 1)
    {
    }

    LogicalResult matchAndRewrite(AffineForOp op,
                                    PatternRewriter& rewriter) const override
    {
        return loopUnrollFull(op);
    }
};

// A pass that invokes the pattern rewrite engine.
struct AffineFullUnrollPatternRewrite
    : impl::AffineFullUnrollPatternRewriteBase<AffineFullUnrollPatternRewrite>
{
    using AffineFullUnrollPatternRewriteBase::
        AffineFullUnrollPatternRewriteBase;
    void runOnOperation() final
    {
        mlir::RewritePatternSet patterns(&getContext());
        patterns.add<AffineFullUnrollPattern>(&getContext());
        // One could use GreedyRewriteConfig here to slightly tweak the behavior
        // of the pattern application.
        (void) applyPatternsGreedily(getOperation(),
                                std::move(patterns));
    }
};

}  // namespace mlir::tutorial