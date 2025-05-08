#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYTYPES_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYTYPES_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir-tutorial/Dialect/Poly/PolyDialect.td"

// A base class for all types in this dialect
class Poly_Type<string name, string typeMnemonic> : TypeDef<PolyDialect, name> {
    let mnemonic = typeMnemonic;  
}

// `def` defines actual types, it generates C++ code
def Polynomial : Poly_Type<"Polynomial", "poly"> {
    let summary = "A polynomial with u32 coefficients";

    let description = [{
        A type for polynomials with integer coefficients in a single-variable polynomial ring.
    }];

    let parameters = (ins "int":$degreeBound);
    let assemblyFormat = "`<` $degreeBound `>`";
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYTYPES_TD_