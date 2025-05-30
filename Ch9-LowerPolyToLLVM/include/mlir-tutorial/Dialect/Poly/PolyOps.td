#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_

include "mlir-tutorial/Dialect/Poly/PolyDialect.td"
include "mlir-tutorial/Dialect/Poly/PolyTypes.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

// Type constaint for poly binop arguments: polys, vectors/tensors of polys
def PolyOrContainer: TypeOrValueSemanticsContainer<Polynomial, "poly-or-container">;

// Inject verification that all integer-like arguments are 32-bit
def Has32BitArguments : NativeOpTrait<"Has32BitArguments"> {
    let cppNamespace = "::mlir::tutorial::poly";
}

// In Ch5 the trait is SameOperandsAndResultElementType,
// enable poly.add to have a mixed poly and tensor-of-poly input semantics. 
class Poly_BinOp<string mnemonic> : Op<PolyDialect, mnemonic, [Pure, ElementwiseMappable, SameOperandsAndResultType]> {
    let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs); 
    let results = (outs PolyOrContainer:$output);
    // With a type inference hook present, we can simplify the operation’s assembly format, 
    // the type need only be specified once instead of three times (type, type) -> type.
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))";   
    let hasFolder = 1;
    let hasCanonicalizer = 1;
}

def Poly_AddOp : Poly_BinOp<"add">{
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

def Poly_ToTensorOp : Op<PolyDialect, "to_tensor", [Pure]> {
    let summary = "Converts a polynomial to a tensor of its integer coefficients.";
    let arguments = (ins Polynomial:$input);
    let results = (outs TensorOf<[AnyInteger]>:$output);
    let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($output)";
}

def IntOrComplex : AnyTypeOf<[AnyInteger, AnyComplex]>;

def Poly_EvalOp : Op<PolyDialect, "eval", [AllTypesMatch<["point", "output"]>, Has32BitArguments]> {
    let summary = "Evaluates a Polynomial at a given input value.";
    let arguments = (ins Polynomial:$input, IntOrComplex:$point);
    let results = (outs IntOrComplex:$output);
    let assemblyFormat = "$input `,` $point attr-dict `:` `(` type($input) `,` type($point) `)` `->` type($output)";
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def Poly_ConstantOp : Op<PolyDialect, "constant", [Pure, ConstantLike]> {
    let summary = "Define a constant polynomial via an attribute.";
    let arguments = (ins AnyIntElementsAttr:$coefficients);
    let results = (outs Polynomial:$output);
    let assemblyFormat = "$coefficients attr-dict `:` type($output)";
    let hasFolder = 1;
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYOPS_TD_