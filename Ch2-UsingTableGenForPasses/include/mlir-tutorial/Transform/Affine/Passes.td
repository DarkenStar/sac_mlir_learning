#ifndef LIB_TRANSFORM_AFFINE_PASSES_TD_
#define LIB_TRANSFORM_AFFINE_PASSES_TD_

include "mlir/Pass/PassBase.td"

// 'def' keyword here specifically implies 
// we’re instantiating this thing in a way that 
// the codegen tool should see and generate real code for. 
def AffineFullUnroll : Pass<"affine-full-unroll"> {
    let summary = "Fully unroll all affine loops";
    let description = [{
        Fully unroll all affine loops.
    }];
    let dependentDialects = ["mlir::affine::AffineDialect"];
}

def AffineFullUnrollPatternRewrite : Pass<"affine-full-unroll-rewrite"> {
    let summary = "Fully unroll all affine loops using the pattern rewrite engine";
    let description = [{
        Fully unroll all affine loops using the pattern rewrite engine.
    }];
    let dependentDialects = ["mlir::affine::AffineDialect"];
}
#endif // LIB_TRANSFORM_AFFINE_PASSES_TD_