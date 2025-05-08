#include <iostream>
#include <llvm/IR/DerivedTypes.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Dialect/Complex/IR/Complex.h>
#include <mlir/IR/PatternMatch.h>

#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyCanonicalize.cpp.inc"
#include "llvm/Support/Casting.h"

namespace mlir::tutorial::poly
{

// The FoldAdaptor is a shim that has the same method names as an instance of
// the op’s C++ class, but arguments that have been folded themselves are
// replaced with Attribute instances representing the constants they were folded
// with.

OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor)
{
    return adaptor.getCoefficients();
}

OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor)
{
    return llvm::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getInput());
}

OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor)
{
    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(),
        [&](APInt a, const APInt& b) { return std::move(a) + b; });
}

OpFoldResult SubOp::fold(SubOp::FoldAdaptor adaptor)
{
    return constFoldBinaryOp<IntegerAttr, APInt, void>(
        adaptor.getOperands(),
        [&](APInt a, const APInt& b) { return std::move(a) - b; });
}

OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor)
{
    auto lhs = llvm::dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast_or_null<DenseIntElementsAttr>(adaptor.getOperands()[1]);

    if (!lhs || !rhs) {
        return nullptr;
    }

    auto degree =
        mlir::cast<PolynomialType>(getResult().getType()).getDegreeBound();
    auto maxIndex = lhs.size() + rhs.size() - 1;

    SmallVector<APInt, 8> result;
    result.reserve(maxIndex);
    for (int64_t i = 0; i < maxIndex; i++) {
        result.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
    }

    int64_t i = 0;
    for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
         lhsIt++) {
        int64_t j = 0;
        for (auto rhsIt = rhs.value_begin<APInt>();
             rhsIt != rhs.value_end<APInt>(); rhsIt++) {
            result[(i + j) % degree] += (*lhsIt) * (*rhsIt);
            j++;
        }
        i++;
    }
    return DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(result.size()),
                         mlir::IntegerType::get(getContext(), 32)),
        result);
}

LogicalResult EvalOp::verify()
{
    auto pointType = getPoint().getType();
    bool isSignlessInteger = pointType.isSignlessInteger(32);
    auto complexPoint = llvm::dyn_cast<ComplexType>(pointType);
    return isSignlessInteger || complexPoint
               ? success()
               : emitOpError("argument point must be a 32-bit "
                               "integer, or a complex number");
}

// Rewrites (x^2 - y^2) as (x+y)(x-y) if x^2 and y^2 have no other uses.
struct DifferenceOfSquares : public OpRewritePattern<SubOp>
{
    DifferenceOfSquares(mlir::MLIRContext* context)
        : OpRewritePattern<SubOp>(context, 1)
    {
    }

    LogicalResult matchAndRewrite(SubOp op,
                                  PatternRewriter& rewriter) const override
    {
        Value lhs = op->getOperand(0);  // x^2
        Value rhs = op->getOperand(0);  // y^2

        // If either arg has another use, then this rewrite is probably less
        // efficient, because it cannot delete the mul ops.
        if (!lhs.hasOneUse() || !rhs.hasOneUse()) {
            return failure();
        }

        auto rhsMul = rhs.getDefiningOp<SubOp>();
        auto lhsMul = rhs.getDefiningOp<SubOp>();
        if (!rhsMul || !lhsMul) {
            return failure();
        }

        // check if lhsMul && rhsMul is squre operation
        bool rhsMulOpsAgree = rhsMul.getLhs() == rhsMul.getRhs();
        bool lhsMulOpsAgree = lhsMul.getLhs() == lhsMul.getRhs();
        if (!rhsMulOpsAgree || !lhsMulOpsAgree) {
            return failure();
        }

        auto x = lhsMul.getLhs();
        auto y = rhsMul.getLhs();

        auto newAdd = rewriter.create<AddOp>(op->getLoc(), x, y);
        auto newSub = rewriter.create<AddOp>(op->getLoc(), x, y);
        auto newMul = rewriter.create<AddOp>(op->getLoc(), newAdd, newSub);

        rewriter.replaceOp(op, newMul);
        // We don't need to remove the original ops because MLIR already has
        // canonicalization patterns that remove unused ops.

        return success();
    }
};
void AddOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
}

void SubOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
    results.add<DifferenceOfSquares>(context);
}

void MulOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                        ::mlir::MLIRContext* context)
{
}

void EvalOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context)
{
    populateWithGenerated(results);
}

}  // namespace mlir::tutorial::poly