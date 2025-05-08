#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h> 
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "mlir-tutorial/Dialect/Poly/PolyDialect.hpp"
#include "mlir-tutorial/Transform/Affine/Passes.h"
#include "mlir-tutorial/Conversion/PolyToStandard/PolyToStandard.hpp"

int main(int argc, char** argv)
{
    mlir::DialectRegistry registry;
    registry.insert<mlir::tutorial::poly::PolyDialect>();
    mlir::registerAllDialects(registry);
    mlir::registerAllPasses();

    mlir::tutorial::registerAffinePasses();
    mlir::tutorial::poly::registerPolyToStandardPasses();
    
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}