#include <cstdint>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "mlir-tutorial/Transform/Arith/MulToAdd.hpp"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir::tutorial
{

using arith::AddIOp;
using arith::ConstantOp;
using arith::MulIOp;

// Replace y = C*x with y = C/2*x + C/2*x, when C is a power of 2,
// otherwise do nothing
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp>
{
    PowerOfTwoExpand(MLIRContext* context)
        : OpRewritePattern<MulIOp>(context, 2)
    {
    }

    LogicalResult matchAndRewrite(MulIOp op, PatternRewriter& rewriter) const override
    {
        // Value is as reference to the operands of the operation,
        // represented as a abstraction of the SSA value.
        // It's not the actual value of the Operand.
        Value lhs = op->getOperand(0);

        // canonicalization patterns ensure the constant is on the right, if
        // there is a constant
        // https://mlir.llvm.org/docs/Canonicalization/#globally-applied-rules
        Value rhs = op.getOperand(1);

        // getDefiningOp take the type you want as output as a template parameter
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();  
        if (!rhsDefiningOp) {
            return failure();
        }

        int64_t value = rhsDefiningOp.value();
        bool is_power_of_two = (value & (value - 1)) == 0;

        if (!is_power_of_two) {
            return failure();
        }

        auto newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp->getLoc(),
            rewriter.getIntegerAttr(rhs.getType(), value / 2));
        auto newMul = rewriter.create<MulIOp>(op->getLoc(), lhs, newConstant);
        auto newAdd = rewriter.create<AddIOp>(op->getLoc(), newMul,newMul);

        rewriter.replaceOp(op, newAdd);
        rewriter.eraseOp(rhsDefiningOp);

        return success();
    }
};

struct PeelFromMul : public OpRewritePattern<MulIOp>
{
    PeelFromMul(MLIRContext* context) : OpRewritePattern<MulIOp>(context, 1)
    {
    }

    LogicalResult matchAndRewrite(MulIOp op,
                                  PatternRewriter& rewriter) const override
    {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        if (!rhsDefiningOp) {
            return failure();
        }

        int64_t value = rhsDefiningOp.value();

        // We are guaranteed `value` is not a power of two, because the
        // greedy rewrite engine ensures the PowerOfTwoExpand pattern is run
        // first, since it has higher benefit.

        auto newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp->getLoc(),
            rewriter.getIntegerAttr(rhs.getType(), value - 1));
        auto newMul = rewriter.create<MulIOp>(op->getLoc(), lhs, newConstant);
        auto newAdd = rewriter.create<AddIOp>(op->getLoc(), newMul, lhs);

        rewriter.replaceOp(op, newAdd);
        rewriter.eraseOp(rhsDefiningOp);

        return success();
    }
};

void MulToAddPass::runOnOperation()
{
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void) applyPatternsGreedily(getOperation(), std::move(patterns));
}
}  // namespace mlir::tutorial
