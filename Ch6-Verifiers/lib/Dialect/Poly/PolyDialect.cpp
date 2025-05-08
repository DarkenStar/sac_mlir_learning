#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"  // Or PolyDialect.hpp
#include "mlir-tutorial/Dialect/Poly/PolyOps.hpp"      // Or PolyOps.hpp
#include "mlir-tutorial/Dialect/Poly/PolyTypes.hpp"  // <--- 添加这一行 (或者 .hpp)

#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

// Include the generated dialect definitions
#include "mlir-tutorial/Dialect/Poly/PolyDialect.cpp.inc"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/UseDefLists.h"
#define GET_TYPEDEF_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-tutorial/Dialect/Poly/PolyOps.cpp.inc"

namespace mlir::tutorial::poly
{

void PolyDialect::initialize()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-tutorial/Dialect/Poly/PolyTypes.cpp.inc"
        >();

    addOperations<
#define GET_OP_LIST
#include "mlir-tutorial/Dialect/Poly/PolyOps.cpp.inc"
        >();
}

Operation* PolyDialect::materializeConstant(OpBuilder& builder, Attribute value,
                                            Type type, Location loc)
{
    auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
    if (!coeffs) {
        return nullptr;
    }
    return builder.create<ConstantOp>(loc,type,coeffs);
}
}  // namespace mlir::tutorial::poly