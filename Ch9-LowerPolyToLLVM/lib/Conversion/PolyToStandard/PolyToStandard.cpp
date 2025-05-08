#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/Support/LogicalResult.h"
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/ImplicitLocOpBuilder.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir::tutorial::poly
{

#define GEN_PASS_DEF_POLYTOSTANDARD
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp.inc"

class PolyToStandardTypeConverter : public TypeConverter
{
public:
    PolyToStandardTypeConverter(MLIRContext* ctx)
    {
        addConversion([](Type type) { return type; });
        addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementType = IntegerType::get(
                ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementType);
        });
    }
};

struct ConvertAdd : public OpConversionPattern<AddOp>
{
    ConvertAdd(MLIRContext* context) : OpConversionPattern<AddOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AddOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto addOp = rewriter.create<arith::AddIOp>(
            op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op.getOperation(), addOp);
        return success();
    }
};

struct ConvertSub : public OpConversionPattern<SubOp>
{
    ConvertSub(MLIRContext* context) : OpConversionPattern<SubOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;
    LogicalResult matchAndRewrite(
        SubOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto SubOp = rewriter.create<arith::SubIOp>(
            op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op.getOperation(), SubOp);
        return success();
    }
};

struct ConvertFromTensor : public OpConversionPattern<FromTensorOp>
{
    ConvertFromTensor(MLIRContext* context)
        : OpConversionPattern<FromTensorOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FromTensorOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto resultTensorType = cast<RankedTensorType>(
            typeConverter->convertType(op->getResultTypes()[0]));
        auto resultShape = resultTensorType.getShape()[0];
        auto resultElementType = resultTensorType.getElementType();

        auto inputTensorType = op.getInput().getType();
        auto inputShape = inputTensorType.getShape()[0];

        // Zero pad the tensor if
        // the coeffs' size is less than the polynomial degree.
        ImplicitLocOpBuilder b(op->getLoc(), rewriter);
        auto coeffValue = adaptor.getInput();
        if (inputShape < resultShape) {
            SmallVector<OpFoldResult, 1> low, high;
            low.push_back(rewriter.getIndexAttr(0));
            high.push_back(rewriter.getIndexAttr(resultShape - inputShape));
            coeffValue = b.create<tensor::PadOp>(
                resultTensorType, coeffValue, low, high,
                b.create<arith::ConstantOp>(
                    rewriter.getIntegerAttr(resultElementType, 0)),
                false);
        }

        rewriter.replaceOp(op, coeffValue);
        return success();
    }
};

struct ConvertToTensor : public OpConversionPattern<ToTensorOp>
{
    ConvertToTensor(mlir::MLIRContext* context)
        : OpConversionPattern<ToTensorOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ToTensorOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        rewriter.replaceOp(op, adaptor.getInput());
        return success();
    }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp>
{
    ConvertConstant(mlir::MLIRContext* context)
        : OpConversionPattern<ConstantOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        ConstantOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);
        auto constOp = b.create<arith::ConstantOp>(adaptor.getCoefficients());
        auto fromTensorOp =
            b.create<FromTensorOp>(op.getResult().getType(), constOp);
        rewriter.replaceOp(op, fromTensorOp.getResult());
        return success();
    }
};

struct ConvertMul : public OpConversionPattern<MulOp>
{
    ConvertMul(MLIRContext* context) : OpConversionPattern<MulOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        MulOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto polymulTensorType =
            cast<RankedTensorType>(adaptor.getLhs().getType());
        auto numTerms = polymulTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op->getLoc(), rewriter);

        // create an all-zeroes tensors to store the result
        auto polymulResult = b.create<ConstantOp>(
            polymulTensorType, DenseIntElementsAttr::get(polymulTensorType, 0));

        // loop bounds and step
        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(0));
        auto numTermsOp = b.create<arith::ConstantOp>(b.getIndexType(),
                                                      b.getIndexAttr(numTerms));
        auto step =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));

        auto p0 = adaptor.getLhs();
        auto p1 = adaptor.getRhs();

        auto outerLoop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, ValueRange(polymulResult.getResult()),
            [&](OpBuilder& builder, Location loc, Value P0Index,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op->getLoc(), builder);
                auto innerLoop = b.create<scf::ForOp>(
                    lowerBound, numTermsOp, step, loopState,
                    [&](OpBuilder& builder, Location loc, Value p1Index,
                        ValueRange loopState) {
                        ImplicitLocOpBuilder b(op->getLoc(), builder);
                        auto accumTensor = loopState.front();
                        auto destIndex = b.create<arith::RemUIOp>(
                            b.create<arith::AddIOp>(P0Index, p1Index),
                            numTermsOp);
                        auto mulOp = b.create<arith::MulIOp>(
                            b.create<tensor::ExtractOp>(p0,
                                                        ValueRange(P0Index)),
                            b.create<tensor::ExtractOp>(p1,
                                                        ValueRange(p1Index)));
                        auto result = b.create<arith::AddIOp>(
                            mulOp, b.create<tensor::ExtractOp>(
                                       accumTensor, destIndex.getResult()));
                        auto stored = b.create<tensor::InsertOp>(
                            result, accumTensor, destIndex.getResult());
                        b.create<scf::YieldOp>(stored.getResult());
                    });
                b.create<scf::YieldOp>(innerLoop.getResults());
            });
        rewriter.replaceOp(op, outerLoop.getResult(0));
        return success();
    }
};

struct ConvertEval : public OpConversionPattern<EvalOp>
{
    ConvertEval(MLIRContext* context)
        : mlir::OpConversionPattern<EvalOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        EvalOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto polyTensorType =
            cast<RankedTensorType>(adaptor.getInput().getType());
        auto numTerms = polyTensorType.getShape()[0];
        ImplicitLocOpBuilder b(op.getLoc(), rewriter);

        auto lowerBound =
            b.create<arith::ConstantOp>(b.getIndexType(), b.getIndexAttr(1));
        auto numTermsOp = b.create<arith::ConstantOp>(
            b.getIndexType(), b.getIndexAttr(numTerms + 1));
        auto step = lowerBound;

        auto poly = adaptor.getInput();
        auto point = adaptor.getPoint();

        auto accum =
            b.create<arith::ConstantOp>(b.getI32Type(), b.getI32IntegerAttr(0));
        auto loop = b.create<scf::ForOp>(
            lowerBound, numTermsOp, step, accum->getResults(),
            [&](OpBuilder builder, Location loc, Value loopIndex,
                ValueRange loopState) {
                ImplicitLocOpBuilder b(op->getLoc(), builder);
                auto accum = loopState.front();
                auto coeffIndex =
                    b.create<arith::SubIOp>(numTermsOp, loopIndex);
                auto mulOp = b.create<arith::MulIOp>(point, accum);
                auto result = b.create<arith::AddIOp>(
                    mulOp,
                    b.create<tensor::ExtractOp>(poly, coeffIndex.getResult()));
                b.create<scf::YieldOp>(result.getResult());
            });

        rewriter.replaceOp(op, loop.getResult(0));
        return success();
    }
};

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard>
{
    using PolyToStandardBase::PolyToStandardBase;

    void runOnOperation() override
    {
        MLIRContext* context = &getContext();
        auto* module = getOperation();

        ConversionTarget target(*context);
        target.addIllegalDialect<PolyDialect>();

        RewritePatternSet patterns(context);
        PolyToStandardTypeConverter typeConverter(context);
        patterns.add<ConvertAdd, ConvertConstant, ConvertSub, ConvertMul,
                     ConvertEval, ConvertFromTensor, ConvertToTensor>(
            typeConverter, context);

        // Func Op
        populateAnyFunctionOpInterfaceTypeConversionPattern(patterns,
                                                            typeConverter);
        target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
            return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                   typeConverter.isLegal(&op.getBody());
        });

        // ReturnOn Op
        populateReturnOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { return typeConverter.isLegal(op); });

        // Call Op
        populateCallOpTypeConversionPattern(patterns, typeConverter);
        target.addDynamicallyLegalOp<func::CallOp>(
            [&](func::CallOp op) { return typeConverter.isLegal(op); });

        // Branch Op
        populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
        target.markUnknownOpDynamicallyLegal([&](Operation* op) {
            return isNotBranchOpInterfaceOrReturnLikeOp(op) ||
                   isLegalForBranchOpInterfaceTypeConversionPattern(
                       op, typeConverter) ||
                   isLegalForReturnOpTypeConversionPattern(op, typeConverter);
        });

        if (failed(
                applyPartialConversion(module, target, std::move(patterns)))) {
            signalPassFailure();
        }
    }
};
}  // namespace mlir::tutorial::poly