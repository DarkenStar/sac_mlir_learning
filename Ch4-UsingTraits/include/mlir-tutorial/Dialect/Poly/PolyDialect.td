#ifndef MLIR_TUTORIAL_DIALECT_POLY_POLYDIALECT_TD_
#define MLIR_TUTORIAL_DIALECT_POLY_POLYDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def PolyDialect: Dialect {
    let name = "poly";
    let summary = "A dialect for polynomial math";
    let description = [{
        The poly dialect defines types and operations for single-variable
        polynomials over integers.
    }];

    let cppNamespace = "::mlir::tutorial::poly";

    let useDefaultTypePrinterParser = 1;
}

#endif  // MLIR_TUTORIAL_DIALECT_POLY_POLYDIALECT_TD_ 