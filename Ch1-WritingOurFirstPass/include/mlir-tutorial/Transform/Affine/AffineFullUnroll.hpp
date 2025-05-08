#pragma once

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir::tutorial
{

class AffineFullUnrollPass
    : public PassWrapper<AffineFullUnrollPass,
                         // Anchors the pass to operation on function bodies
                         // and provides the getOperation method which returns
                         // the FuncOp being operated on
                         OperationPass<mlir::func::FuncOp>>

{
private:
    // the function that performs the pass logic.
    void runOnOperation() override;

    //  the CLI argument for an mlir-opt-like tool
    StringRef getArgument() const final
    {
        return "affine-full-unroll";
    }

    // the CLI description when running --help on the mlir-opt-like tool.
    StringRef getDescription() const final
    {
        return "Fully unroll all affine loops";
    }
};

class AffineFullUnrollPassAsPatternRewrite
    : public PassWrapper<AffineFullUnrollPass, OperationPass<func::FuncOp>>
{
private:
    void runOnOperation() override;
    StringRef getArgument() const final
    {
        return "affine-full-unroll-rewrite";
    }

    StringRef getDescription() const final
    {
        return "Fully unroll all affine loops using pattern rewrite engine";
    }
};
}  // namespace mlir::tutorial
