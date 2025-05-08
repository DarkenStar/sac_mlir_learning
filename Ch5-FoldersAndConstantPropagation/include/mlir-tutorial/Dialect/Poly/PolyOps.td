#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_

include "mlir-tutorial/Dialect/Poly/PolyDialect.td"
include "mlir-tutorial/Dialect/Poly/PolyTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"
// Type constaint for poly binop arguments: polys, vectors/tensors of polys
def PolyOrContainer: TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemonic> : Op<PolyDialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultElementType]> {
    let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs); 
    let results = (outs PolyOrContainer:$output);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";   
    let hasFolder = 1;
}

def poly_AddOp : Poly_BinOp<"add">{
    let summary = "Addition operation between polynomials.";   
}


def Poly_SubOp : Poly_BinOp<"sub"> {
    let summary = "Subtraction operation between polynomials.";
}

def Poly_MulOp : Poly_BinOp<"mul"> {
    let summary = "Subtraction operation between polynomials.";
}

def Poly_FromTensorOp: Op<PolyDialect, "from_tensor", [Pure]> {
    let summary = "Creates a Polynomial from integer coefficients stored in a tensor.";
    let arguments = (ins TensorOf<[AnyInteger]>:$input);
    let results = (outs Polynomial:$output);
    let assemblyFormat  = "$input attr-dict `:` type($input) `->` type($output)"; 
    let hasFolder = 1;
}

def Poly_EvalOp : Op<PolyDialect, "eval"> {
    let summary = "Evaluates a Polynomial at a given input value.";
    let arguments = (ins Polynomial:$input, AnyInteger:$point);
    let results = (outs AnyInteger:$output);
    let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
}

def Poly_ConstantOp : Op<PolyDialect, "constant", [Pure, ConstantLike]> {
    let summary = "Define a constant polynomial via an attribute.";
    let arguments = (ins AnyIntElementsAttr:$coefficients);
    let results = (outs Polynomial:$output);
    let assemblyFormat = "$coefficients attr-dict `:` type($output)";
    let hasFolder = 1;
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_