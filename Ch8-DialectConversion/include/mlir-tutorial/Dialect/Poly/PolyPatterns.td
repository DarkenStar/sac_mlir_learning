#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYPATTERNS_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYPATTERNS_TD_

include "mlir-tutorial/Dialect/Poly/PolyOps.td"
include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"

def LiftConjThroughEval : Pat<(Poly_EvalOp $f, (ConjOp $z, $fastmath)),
                                (ConjOp (Poly_EvalOp $f, $z), $fastmath)>;

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

def DifferenceOfSquares : Pattern<
    (Poly_SubOp (Poly_MulOp:$lhs $x, $x), (Poly_MulOp:$rhs $y, $y)),
[
    (Poly_AddOp:$sum $x, $y),
    (Poly_SubOp:$diff $x, $y),
    (Poly_MulOp:$res $sum, $diff),
],
[
    (HasOneUse: $lhs), (HasOneUse: $lhs)
]
>;
#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYPATTERNS_TD_
