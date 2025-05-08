#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Affine/LoopUtils.h>
#include <mlir/IR/MLIRContext.h>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir-tutorial/Transform/Affine/AffineFullUnroll.hpp"

namespace mlir::tutorial
{
using ::mlir::affine::AffineForOp;
using ::mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation()
{
    // walk traverse the AST of the FuncOp
    getOperation()->walk([&](affine::AffineForOp op) {
        if (failed(loopUnrollFull(op))) {
            op->emitOpError("unrolling failed");
            signalPassFailure();
        }
    });
}

struct AffineFullUnrollPattern : public OpRewritePattern<affine::AffineForOp>
{
    // Constructor
    AffineFullUnrollPattern(MLIRContext* context)
        : mlir::OpRewritePattern<AffineForOp>(context, 1)
    {
    }

    LogicalResult matchAndRewrite(AffineForOp op,
                                  PatternRewriter& rewriter) const override
    {
        // In a proper OpRewritePattern, the mutations of the IR must go through
        // the PatternRewriter argument
        return loopUnrollFull(op);
    }
};

// A pass that invokes the pattern rewrite engine.
void AffineFullUnrollPassAsPatternRewrite::runOnOperation()
{
    RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    // One could use GreedyRewritingConfig here to slightly tweak the behavior
    // of the patter application
    (void)applyPatternsGreedily(getOperation(), std::move(patterns));
    }
}  // namespace mlir::tutorial
