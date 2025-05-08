#pragma once

#include "mlir/Support/LLVM.h"
#include <mlir/IR/OpDefinition.h>

namespace mlir::tutorial::poly
{

template <typename ConceretType>
class Has32BitArguments
    : public OpTrait::TraitBase<ConceretType, Has32BitArguments>
{
public:
    static LogicalResult verifyTrait(Operation* op)
    {
        for (auto type : op->getOperandTypes()) {
            // OK to skip non-integer Operand types
            if (!type.isIntOrIndex()) {
                continue;
            }

            if (!type.isInteger(32)) {
                return op->emitError()
                       << "requires each numeric operand to be a 32-bit integer";
            }
        }

        return success();
    }
};

}  // namespace mlir::tutorial::poly