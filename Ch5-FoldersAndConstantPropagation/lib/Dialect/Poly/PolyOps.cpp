#include <cstdint>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/CommonFolders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/OpDefinition.h>

#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

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
    return dyn_cast<DenseIntElementsAttr>(adaptor.getInput());
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
    auto lhs = dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[1]);

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
}  // namespace mlir::tutorial::poly